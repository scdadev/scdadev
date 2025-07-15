import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import augmentations

from copy import deepcopy
from pytorch_msssim import ssim

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h
    




class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, log_std_bounds):
        super().__init__()

       
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, 2*action_shape[0]))

        self.log_std_bounds = log_std_bounds
  

        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        mu, log_std = self.policy(h).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
                                                                     1)
        std = log_std.exp()

        dist = utils.SquashedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action, step=10, info=None):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class AGENT:
    ''' SVEA: Stabilized Q-Value Estimation under Augmentation '''
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, init_temperature,
                 log_std_bounds, strong_augs, spda_para, add_clip, clip_step, clip_para):

        self.device = device
        self.strong_aug = strong_augs
        self.critic_target_tau = critic_target_tau
        self.consistency = 1
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.lr = lr
        self.consistency = True
        self.applytransfer = True
        self.conventer_pretrain = True

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim, log_std_bounds).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_shape[0]

        self.auto_converter = utils.Autogenerator().to(device)
        self.jug_encoder = utils.jug_encoder(deepcopy(self.encoder),deepcopy(self.critic.trunk)).to(device)
        for param in self.jug_encoder.parameters():
            param.requires_grad = False
        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)

        self.AC01_optimizer = torch.optim.Adam(self.auto_converter.parameters(), lr=self.lr)

       # data augmentation
        self.aug = augmentations.random_shift(pad=4)
        self.apply_strong_aug = augmentations.compose_augs(strong_augs)

        self.train()
        self.critic_target.train()

        self.dynamic = Dynamics(self.encoder, action_shape[0], feature_dim, hidden_dim, self.critic.trunk).to(device)

        self.dynamic_optimizer = torch.optim.Adam(
            self.dynamic.parameters(),
            lr=3e-4,
            betas=(0.9, 0.999),
        )

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        dist = self.actor(obs)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)


        return action.cpu().detach().numpy()[0]


    def update_critic(self, obs_e, action, reward, discount, next_obs_e, aug_obs_e, step, info=None, mask=None):
        metrics = dict()
        with torch.no_grad():
            dist = self.actor(next_obs_e)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs_e, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (discount * target_V)


        # get current Q estimates
        combined_obs = torch.cat([obs_e, aug_obs_e], dim=0)
        combined_action = torch.cat([action, action], dim=0)
        combined_target_Q = torch.cat([target_Q, target_Q], dim=0)

        combined_Q1, combined_Q2 = self.critic(combined_obs, combined_action, step=step, info=info)
        reg_Q1, aug_Q1 = combined_Q1.chunk(2, dim=0)
        reg_Q2, aug_Q2 = combined_Q2.chunk(2, dim=0)
        reg_Q = torch.cat([reg_Q1, reg_Q2], dim=0)
        aug_Q = torch.cat([aug_Q1, aug_Q2], dim=0)
        
        critic_loss = (F.mse_loss(reg_Q, combined_target_Q) + F.mse_loss(aug_Q, combined_target_Q)) * 0.5

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['reg_critic_q1'] = reg_Q1.mean().item()
        metrics['reg_critic_q2'] = reg_Q2.mean().item()
        metrics['aug_critic_q1'] = aug_Q1.mean().item()
        metrics['aug_critic_q2'] = aug_Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs):
        metrics = dict()

        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = (self.alpha.detach() * log_prob - Q).mean()

        # optimize the actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()


        self.log_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_entropy'] = -log_prob.mean().item()
        metrics['actor_target_entropy'] = self.target_entropy
        metrics['alpha_loss'] = alpha_loss.item()
        metrics['alpha_value'] = self.alpha
        

        return metrics

    def update_dynamic(self, obs, action, reward, next_obs, mask, step=None, L=None):
        with torch.no_grad():
            next_h = self.encoder(next_obs)
            next_h = self.critic.trunk(next_h)
        predict_h = self.dynamic(obs,action)
        dynamic_loss = F.mse_loss(predict_h, next_h)
        

        if self.consistency:
            if 'attmask' in self.strong_aug[0] or 'all' == self.strong_aug[0]:
                aug_obs = self.apply_strong_aug(obs.clone(), mask=mask)
            else:
                aug_obs = self.apply_strong_aug(obs.clone())
                mask = None
            # masked_obs = augmentations.random_overlay(obs.clone())    # apply random_overlay
            masked_predict_h = self.dynamic(aug_obs,action)
            dynamic_loss += 0.5 * F.mse_loss(masked_predict_h, next_h)


        self.dynamic_optimizer.zero_grad()
        dynamic_loss.backward()
        self.dynamic_optimizer.step()

    # Altering to include strong aug
    def update(self, replay_iter, step, info= None):

        if self.conventer_pretrain and self.applytransfer:    # pretrain the converter
            print("converter pretrain started")
            for _ in range(100000):
                batch = next(replay_iter)
                obs, _, _, _, _ = utils.to_torch(
                    batch, self.device)
                obs = self.aug(obs.float())

                if 'attmask' in self.strong_aug[0] or 'all' == self.strong_aug[0]:
                    _obs = obs - obs.mean(dim=0)
                    mask = utils.compute_attribution_mask(_obs, quantile=0.94).long()
                    aug_obs = self.apply_strong_aug(obs.clone(), mask=mask)
                else: 
                    aug_obs = self.apply_strong_aug(obs.clone())
                    mask = None

                aug_obs_gen = self.auto_converter(aug_obs)
                loss_converter_pretrain = F.mse_loss(aug_obs_gen/255.0, aug_obs/255.0)
                loss_ssim = 1 - ssim(aug_obs_gen/255.0, aug_obs/255.0, data_range=1.0, size_average=True)
                loss_rebuilt = loss_converter_pretrain + 0.2*loss_ssim
                # loss_rebuilt = loss_converter_pretrain
                self.AC01_optimizer.zero_grad()
                loss_rebuilt.backward()
                self.AC01_optimizer.step()
            print("converter pretrain finished")

            self.AC02_optimizer = torch.optim.Adam(self.auto_converter.parameters(), lr=self.lr)
            self.conventer_pretrain = False

        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)
        
        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())

        if 'attmask' in self.strong_aug[0] or 'all' == self.strong_aug[0]:
            _obs = obs - obs.mean(dim=0)
            mask = utils.compute_attribution_mask(_obs, quantile=0.94).long()
            aug_obs = self.apply_strong_aug(obs.clone(), mask=mask)
        else:
            aug_obs = self.apply_strong_aug(obs.clone())
            mask = None

        if self.consistency:
            if self.applytransfer:
                aug_obs_gen = self.auto_converter(aug_obs)
                loss_rebuilt = F.mse_loss(aug_obs_gen/255.0, aug_obs/255.0)
                loss_ssim = 1 - ssim(aug_obs_gen/255.0, aug_obs/255.0, data_range=1.0, size_average=True)
                loss_gen1 = loss_rebuilt + 0.2*loss_ssim

                emb = self.jug_encoder(aug_obs_gen)
                loss_gen2 = -torch.trace(torch.cov(emb.T))*0.0001
                # emb = F.normalize(emb, dim=1)    # ------------ or replace with this
                # similarity_matrix = emb @ emb.T    # ------------ cos similarity
                # loss_gen2 = -similarity_matrix.mean()*0.01    # ------------
                ratio = 1
                if abs(loss_gen2)>=2.0*abs(loss_gen1):
                    with torch.no_grad():
                        ratio = abs(loss_gen1*2.0/loss_gen2)
                        loss_gen2 = loss_gen2*ratio
                loss_gen = loss_gen1+loss_gen2
                self.AC02_optimizer.zero_grad()
                loss_gen.backward()
                self.AC02_optimizer.step()

                choices = torch.rand(aug_obs.shape[0], device=self.device) < 1/2
                choices_expanded = choices[:, None, None, None]
                aug_obs_gen = aug_obs_gen.detach()
                aug_obs = torch.where(choices_expanded, aug_obs, aug_obs_gen)

        # encode
        combined_obs = torch.cat([obs, aug_obs], dim=0)
        obs_e, aug_obs_e = self.encoder(combined_obs).chunk(2, dim=0)
        with torch.no_grad():
            next_obs_e = self.encoder(next_obs)

        metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic(obs_e, action, reward, discount, next_obs_e, aug_obs_e, step=step, info=info, mask=mask))

        # update actor
        metrics.update(self.update_actor(obs_e.detach()))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)
        
        self.update_dynamic(obs, action, reward, next_obs, mask, step)

        if self.applytransfer and step % 2500 == 0:
            utils.soft_update_params(self.encoder, self.jug_encoder.encoder1, 1)
            utils.soft_update_params(self.critic.trunk, self.jug_encoder.trunk, 1)

        return metrics
    
class Dynamics(nn.Module):
    def __init__(self, encoder, action_shape, feature_dim, hidden_dim, truck):
        super().__init__()
        self.encoder = encoder
        self.trunk = truck
        self.mlp = nn.Sequential(
            nn.Linear(action_shape + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.apply(weight_init)


    def forward(self, x, action):
        h = self.encoder(x)
        h = self.trunk(h)
        joint_h = torch.cat([h, action], dim=1)

        return self.mlp(joint_h)

    
def weight_init(m):
    """Custom weight init for Conv2D and Linear layers"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)