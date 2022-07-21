import numpy as np
import torch
import joblib
import torch.nn as nn
from torch.distributions import Categorical
from utils import *


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + (delta ** 0.5) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, size):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            # nn.BatchNorm2d(size),
            layer_init(nn.Conv2d(size, size, kernel_size=(3, 3), padding=(1, 1), padding_mode='circular')),
            nn.ReLU(),
            # nn.BatchNorm2d(size),
            layer_init(nn.Conv2d(size, size, kernel_size=(3, 3), padding=(1, 1), padding_mode='circular')),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x) + x


class DownsampleBlock(nn.Module):
    def __init__(self, size):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(size, size, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), padding_mode='circular')),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class RunningMeanVar(nn.Module):
    def __init__(self, size):
        super(RunningMeanVar, self).__init__()
        self.mean = nn.Parameter(torch.zeros(size), requires_grad=False)
        self.var = nn.Parameter(torch.ones(size), requires_grad=False)
        self.count = nn.Parameter(torch.tensor(1e-4), requires_grad=False)

    def reset(self):
        self.mean.data = torch.zeros_like(self.mean.data)
        self.var.data = torch.ones_like(self.var.data)
        self.count.data = torch.ones_like(self.count.data) * 1e-4

    def update(self, batch, dim):
        batch_mean, batch_var, batch_count = batch.mean(dim=dim), batch.var(dim=dim), batch.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        self.mean.data, self.var.data, self.count.data = new_mean, new_var, new_count


class ActorCritic(nn.Module):
    def __init__(self, hidden_size, n_blocks):
        super(ActorCritic, self).__init__()

        self.policy_state_mean_var = RunningMeanVar(POLICY_INPUT_SIZE)
        self.value_state_mean_var = RunningMeanVar(VALUE_INPUT_SIZE)
        self.return_mean_var = RunningMeanVar(1)

        self.actor = nn.Sequential(
            layer_init(nn.Conv2d(POLICY_INPUT_SIZE, hidden_size, kernel_size=(3, 3), padding=(1, 1), padding_mode='circular')),
            nn.ReLU(),
            *[ResidualBlock(hidden_size) for _ in range(n_blocks)],
            layer_init(nn.Conv2d(hidden_size, ACTION_SIZE, kernel_size=(1, 1)), std=0.01)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(VALUE_INPUT_SIZE, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1), std=1)
        )

    def forward_value(self, x):
        b, c, h, w = x.shape
        # normalize input
        x = (x - self.value_state_mean_var.mean.view(1, c, 1, 1)) / torch.sqrt((self.value_state_mean_var.var.view(1, c, 1, 1) + 1e-8))
        value = self.critic(x.mean(dim=(2,3)))
        value = value.squeeze(-1)
        return value

    def forward_policy(self, x, action=None, mask=None):
        b, c, h, w = x.shape
        # normalize input
        x = (x - self.policy_state_mean_var.mean.view(1, c, 1, 1)) / torch.sqrt((self.policy_state_mean_var.var.view(1, c, 1, 1) + 1e-8))
        # cnn
        x = self.actor(x)
        # mask
        if mask is not None:
            mask = mask.bool()  # [b, ACTION_SIZE, 21, 21]
            assert x.shape == mask.shape
            x = torch.where(mask, x, torch.tensor(-1e8, dtype=x.dtype, device=x.device))  # -1e8 if using inf_mask
            inf_mask = (mask.sum(dim=(1,2,3), keepdim=True) == 0) | mask
            x = torch.where(inf_mask, x, torch.tensor(-np.inf, dtype=x.dtype, device=x.device))
        # get output
        dist = Categorical(logits=x.view(b, -1))  # [b, ACTION_SIZEx21x21]
        action = dist.sample() if action is None else action.to(dtype=torch.int64)  # [b]
        out = {
            'logit_map': x,
            'dist': dist,
            'action': action,
            'log_prob': dist.log_prob(action),
            'entropy': dist.entropy()
        }

        return out