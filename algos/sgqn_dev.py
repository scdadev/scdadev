import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import augmentations

from utils import (
    compute_attribution,
    compute_attribution_mask,
    make_attribution_pred_grid,
    make_obs_grid,
    make_obs_grad_grid,
)
import random


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
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.consistency = True
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

        self.attribution_predictor = AttributionPredictor(action_shape[0], self.encoder, self.critic.trunk).to(device)
        self.quantile = 0.9

        self.aux_optimizer = torch.optim.Adam(
            self.attribution_predictor.parameters(),
            lr=3e-4,
            betas=(0.9, 0.999),
        )

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
       # data augmentation
        self.aug = augmentations.random_shift(pad=4)
        self.apply_strong_aug = augmentations.compose_augs(strong_augs)

        self.train()
        self.critic_target.train()

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


    def update_critic(self, obs,obs_e, action, reward, discount, next_obs,next_obs_e, step, aug_obs=None, info=None):
        metrics = dict()
        with torch.no_grad():
            dist = self.actor(next_obs_e)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs_e, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (discount * target_V)


        Q1, Q2 = self.critic(obs_e, action, step=step, info=info)
        
        critic_loss = (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)) * 0.5

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['reg_critic_q1'] = Q1.mean().item()
        metrics['reg_critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        if self.consistency:
            obs_grad = compute_attribution(self.encoder,self.critic,obs,action.detach())
            mask = compute_attribution_mask(obs_grad,self.quantile)
            masked_obs = obs*mask
            masked_obs[mask<1] = random.uniform(obs.view(-1).min(),obs.view(-1).max()) 
            masked_Q1,masked_Q2 = self.critic(self.encoder(masked_obs),action)
            critic_loss += 0.5 *(F.mse_loss(Q1,masked_Q1) + F.mse_loss(Q2,masked_Q2))

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
    
    def update_aux(self, obs, action, obs_grad, mask, step=None, L=None):
        mask = compute_attribution_mask(obs_grad,self.quantile)

        if 'attmask' in self.strong_aug[0] or 'all' == self.strong_aug[0]:
            _obs = obs - obs.mean(dim=0)
            mask_att = utils.compute_attribution_mask(_obs, quantile=0.9).long()
            s_tilde = self.apply_strong_aug(obs.clone(), mask=mask_att)
        else: 
            s_tilde = self.apply_strong_aug(obs.clone())
        self.aux_optimizer.zero_grad()
        pred_attrib, aux_loss = self.compute_attribution_loss(s_tilde,action, mask)
        aux_loss.backward()
        self.aux_optimizer.step()

    def compute_attribution_loss(self, obs,action, mask):    # obs: random_overlay
        mask = mask.float()
        attrib = self.attribution_predictor(obs.detach(),action.detach())
        aux_loss = F.binary_cross_entropy_with_logits(attrib, mask.detach())
        return attrib, aux_loss

    # Altering to include strong aug
    def update(self, replay_iter, step, info= None):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)
        
        # augment
        obs = self.aug(obs.float())
        # aug_obs = self.apply_strong_aug(obs.clone())
        next_obs = self.aug(next_obs.float())

        # encode
        # combined_obs = torch.cat([obs, aug_obs], dim=0)
        # obs, aug_obs = self.encoder(combined_obs).chunk(2, dim=0)
        obs_e = self.encoder(obs)
        with torch.no_grad():
            next_obs_e = self.encoder(next_obs)

        metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic(obs,obs_e, action, reward, discount, next_obs,next_obs_e, step=step, aug_obs=None, info=info))

        obs_grad = compute_attribution(self.encoder,self.critic,obs,action.detach())
        mask = compute_attribution_mask(obs_grad,self.quantile)
        
        # update actor
        metrics.update(self.update_actor(obs_e.detach()))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                self.critic_target_tau)
        
        self.update_aux(obs, action, obs_grad, mask, step)

        return metrics
    

class AttributionDecoder(nn.Module):
    def __init__(self,action_shape, emb_dim=100) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features=emb_dim+action_shape, out_features=14112)
        self.conv1 = nn.Conv2d(
            in_channels=32, out_channels=128, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=9, kernel_size=3, padding=1)

    def forward(self, x, action):
        x = torch.cat([x,action],dim=1)
        x = self.proj(x).view(-1, 32, 21, 21)
        x = self.relu(x)
        x = self.conv1(x)
        x = F.upsample(x, scale_factor=2)
        x = self.relu(x)
        x = self.conv2(x)
        x = F.upsample(x, scale_factor=2)
        x = self.relu(x)
        x = self.conv3(x)
        return x

class AttributionPredictor(nn.Module):
    def __init__(self, action_shape,encoder,trunk,emb_dim=100):
        super().__init__()
        self.encoder = encoder
        self.trunk = trunk
        self.decoder = AttributionDecoder(action_shape,emb_dim)
        self.features_decoder = nn.Sequential(
            nn.Linear(emb_dim, 256), nn.ReLU(), nn.Linear(256, emb_dim)
        )

    def forward(self, x,action):    # x:random_overlay obs
        x = self.encoder(x)
        x = self.trunk(x)
        return self.decoder(x,action)