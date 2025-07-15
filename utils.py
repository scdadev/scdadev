
import random
import time
import numpy as np
import torch


from torch import distributions as pyd
import torch.nn as nn
import torch.nn.functional as F
import math
from captum.attr import GuidedBackprop, GuidedGradCam
from torchvision.utils import make_grid

# ------------------------------------------- Train Utils -----------------------------------------------#

class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def add_aug_directory(dirs):
    try:
        import augmentations
        augmentations.random_overlay.places_dirs = dirs
    except:
        pass



class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until    # 1000000
        self._action_repeat = action_repeat    # 4

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


# ----------------------------------------- Algos Utils ----------------------------------------------------------#
    


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

def compute_attribution_mask(obs_grad, quantile=0.95):
    mask = []
    for i in [0, 3, 6]:
        attributions = obs_grad[:, i : i + 3].abs().max(dim=1)[0]
        q = torch.quantile(attributions.flatten(1), quantile, 1)
        mask.append((attributions >= q[:, None, None]).unsqueeze(1).repeat(1, 3, 1, 1))
    return torch.cat(mask, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Autogenerator(nn.Module):
    def __init__(self):
        super(Autogenerator, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.middle = ResidualBlock(32)

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32, 9, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)    # -> [B,16,42,42]
        e2 = self.enc2(e1)   # -> [B,32,21,21]
        z = self.middle(e2)  # -> [B,32,21,21]

        d1 = self.dec1(z)    # -> [B,16,42,42]
        d1_cat = torch.cat([d1, e1], dim=1)  # -> [B,32,42,42]
        out = self.dec2(d1_cat) * 255.0      # -> [B,9,84,84]
        return out
    
class jug_encoder(nn.Module):
    def __init__(self, encoder1, trunk):
        super(jug_encoder, self).__init__()
        self.encoder1 = encoder1
        self.trunk = trunk
    
    def forward(self, x):
        return self.trunk(self.encoder1(x))

# ----------------------------------------- sgqn -----------------------------------------#
def compute_attribution(encoder, critic, obs, action=None,method="guided_backprop"):
    if method == "guided_backprop":
        return compute_guided_backprop(obs, action, encoder, critic)    # model: critic
    if method == 'guided_gradcam':
        return compute_guided_gradcam(obs,action,encoder, critic)
    return compute_vanilla_grad(encoder, critic, obs, action)

def compute_guided_backprop(obs, action, encoder, critic):    # model: critic
    model = ModelWrapper(encoder, critic, action=action)
    gbp = GuidedBackprop(model)
    attribution = gbp.attribute(obs)
    return attribution

def compute_guided_gradcam(obs, action, encoder, critic):
    obs.requires_grad_()
    obs.retain_grad()
    model = ModelWrapper(encoder, critic, action=action)
    gbp = GuidedGradCam(model,layer=model.model.encoder.head_cnn.layers)
    attribution = gbp.attribute(obs,attribute_to_layer_input=True)
    return attribution

def compute_vanilla_grad(critic_target, obs, action):
    obs.requires_grad_()
    obs.retain_grad()
    q, q2 = critic_target(obs, action.detach())
    q.sum().backward()
    return obs.grad

class ModelWrapper(torch.nn.Module):
    def __init__(self, encoder, critic, action=None):
        super(ModelWrapper, self).__init__()
        self.encoder = encoder
        self.critic = critic
        self.action = action

    def forward(self, obs):
        if self.action is None:
            return self.critic(self.encoder(obs))[0]
        return self.critic(self.encoder(obs), self.action)[0]
    
def make_obs_grid(obs, n=4):
    sample = []
    for i in range(n):
        for j in range(0, 9, 3):
            sample.append(obs[i, j : j + 3].unsqueeze(0))
    sample = torch.cat(sample, 0)
    return make_grid(sample, nrow=3) / 255.0


def make_attribution_pred_grid(attribution_pred, n=4):
    return make_grid(attribution_pred[:n], nrow=1)


def make_obs_grad_grid(obs_grad, n=4):
    sample = []
    for i in range(n):
        for j in range(0, 9, 3):
            channel_attribution, _ = torch.max(obs_grad[i, j : j + 3], dim=0)
            sample.append(channel_attribution[(None,) * 2] / channel_attribution.max())
    sample = torch.cat(sample, 0)
    q = torch.quantile(sample.flatten(1), 0.97, 1)
    sample[sample <= q[:, None, None, None]] = 0
    return make_grid(sample, nrow=3)