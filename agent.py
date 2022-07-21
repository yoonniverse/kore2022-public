import torch.nn as nn
import torch.optim as optim
import copy
import joblib
from kaggle_environments.envs.kore_fleets.helpers import Board

from net import ActorCritic
from processors import process_obs, process_action
from utils import *


class Agent(nn.Module):
    def __init__(self, cfg):
        super(Agent, self).__init__()
        self.cfg = cfg
        self.device = 'cpu'

        self.ge = 0
        self.actor_critic = ActorCritic(self.cfg.hidden_size, self.cfg.n_blocks)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.cfg.lr)
        self.state_dicts_history = np.array([copy.deepcopy(self.state_dict())])
        self.state_dicts_priority = np.array([1], dtype=np.float32)
        self.eval()

        # trajectory buffer
        self.trajectory = None
        self.reset_trajectory()

    def set_device(self, device):
        device = torch.device(device)
        self.device = device
        self.to(device=device)
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    def reset_trajectory(self):
        self.trajectory = {
            'action': [],
            'log_prob': [],
            'value': [],
            'policy_state': [],
            'value_state': [],
            'step': [],
            'done': [],
            'reward': [],
            'logit_map': [],
            'action_mask': []
        }

    def act(self, obs, env_config, greedy=False):
        # process observation
        policy_states, value_state, masks = process_obs(obs, env_config)
        # forward_value
        value_state_tensor = torch.from_numpy(value_state).to(device=self.device).unsqueeze(0)
        value = self.actor_critic.forward_value(value_state_tensor)
        # unnormalize value
        value = (value * torch.sqrt(self.actor_critic.return_mean_var.var) + self.actor_critic.return_mean_var.mean).squeeze().item()
        self.trajectory['value_state'].append(value_state)
        self.trajectory['value'].append(value)
        if policy_states is None:  # current player has no shipyards
            return {}
        # forward_policy
        policy_state_tensor = torch.from_numpy(policy_states).to(device=self.device)  # [n_shipyards, STATE_SIZE+2, BOARD_SIZE, BOARD_SIZE]
        mask_tensor = torch.from_numpy(masks).to(device=self.device)
        self.actor_critic.eval()
        with torch.no_grad():
            out = self.actor_critic.forward_policy(policy_state_tensor, mask=mask_tensor)
        self.actor_critic.train()
        # process action
        if greedy:
            action = out['dist'].probs.argmax(dim=1).cpu().numpy()
        else:
            action = out['action'].cpu().numpy()
        processed_action = process_action(action, obs, env_config)
        # save at trajectory
        assert action.ndim == 1
        self.trajectory['action'] += [x for x in action]
        assert out['log_prob'].ndim == 1
        self.trajectory['log_prob'] += [x for x in out['log_prob'].cpu().numpy()]
        assert out['logit_map'].ndim == 4
        self.trajectory['logit_map'] += [x for x in out['logit_map'].cpu().numpy()]
        self.trajectory['policy_state'] += [x for x in policy_states]
        self.trajectory['action_mask'] += [x for x in masks]
        self.trajectory['step'] += [obs['step'] for _ in range(len(policy_states))]
        # debug
        # for i, (k, v) in enumerate(processed_action.items()):
        #     if len(v) == 0:
        #         print('no action!!')
        #         joblib.dump((obs, env_config, masks, out, processed_action), 'error.jl')
        return processed_action

    def learn(self, e):
        """
        e: batch sampled from ppo buffer
        """
        logs = {
            'actor_loss': 0,
            'critic_loss': 0,
            'entropy_loss': 0,
            'approx_kl': 0,
            'clip_frac': 0,
            'max_abs_grad': 0
        }

        for k in e.keys():
            e[k] = e[k].to(device=self.device)
            if torch.isnan(e[k]).any() or torch.isinf(e[k]).any():
                joblib.dump(e, 'error.jl')
                exit()
        try:
            out = self.actor_critic.forward_policy(e['policy_state'], action=e['action'], mask=e['action_mask'])
        except ValueError as error:
            print(error)
            for k in e.keys():
                e[k] = e[k].to(device=self.device)
                joblib.dump(e, 'error.jl')
            return logs

        value = self.actor_critic.forward_value(e['value_state'])
        log_ratio = out['log_prob'] - e['log_prob']
        ratio = log_ratio.exp()

        approx_kl = ((ratio - 1) - log_ratio).mean().item()
        clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean().item()
        
        surr1 = ratio * e['advantage']
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef) * e['advantage']
        actor_loss = -torch.min(surr1, surr2).mean()

        # normalize return
        e['return'] = (e['return'] - self.actor_critic.return_mean_var.mean) / torch.sqrt(self.actor_critic.return_mean_var.var)
        critic_loss = (e['return'] - value).pow(2).mean()

        entropy_loss = -out['entropy'].mean()

        pl_coef = 1 if self.ge > self.cfg.only_value_ge else 0
        loss = pl_coef * actor_loss + self.cfg.vf_coef * critic_loss + pl_coef * self.cfg.ent_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        max_abs_grad = 0
        for p in self.actor_critic.parameters():
            if p.grad is not None:
                abs_grad = abs(p.grad.max().item())
                if abs_grad > max_abs_grad:
                    max_abs_grad = abs_grad
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()

        logs['actor_loss'] = actor_loss.item()
        logs['critic_loss'] = critic_loss.item()
        logs['entropy_loss'] = entropy_loss.item()
        logs['approx_kl'] = approx_kl
        logs['clip_frac'] = clip_frac
        logs['max_abs_grad'] = max_abs_grad

        return loss.item(), logs

    def learn_imitation(self, policy_inputs, mask_inputs, action_inputs):
        policy_inputs = policy_inputs.to(self.device)
        mask_inputs = mask_inputs.to(self.device)
        action_inputs = action_inputs.to(self.device)
        out = self.actor_critic.forward_policy(policy_inputs, action=action_inputs, mask=mask_inputs)
        # joblib.dump((policy_inputs, mask_inputs, action_inputs, out), 'tmp.jl')
        # exit()
        loss = -out['log_prob'].mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            accuracy = (out['dist'].probs.argmax(1) == action_inputs).float().mean()
        return loss.item(), accuracy.item()

    def save(self, path):
        torch.save({
            'cfg': self.cfg,
            'state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ge': self.ge,
            'state_dicts_history': self.state_dicts_history,
            'state_dicts_priority': self.state_dicts_priority
        }, path)

    def load(self, path, load_optimizer):
        ckp = torch.load(path, map_location='cpu')
        self.load_state_dict(ckp['state_dict'])
        if load_optimizer:
            self.optimizer.load_state_dict(ckp['optimizer_state_dict'])
        self.ge = ckp['ge']
        self.state_dicts_history = ckp['state_dicts_history']
        self.state_dicts_priority = ckp['state_dicts_priority']

    def sort_state_dicts_history(self):
        idx = np.argsort(self.state_dicts_priority)
        self.state_dicts_history = self.state_dicts_history[idx]
        self.state_dicts_priority = self.state_dicts_priority[idx]

    def append_state_dicts_history(self, state_dict):
        self.state_dicts_history = np.append(self.state_dicts_history, copy.deepcopy(state_dict))
        self.state_dicts_priority = np.append(self.state_dicts_priority, 1)
        if len(self.state_dicts_history) > self.cfg.len_state_dicts_queue:
            self.state_dicts_history = self.state_dicts_history[1:]
            self.state_dicts_priority = self.state_dicts_priority[1:]
        # self.sort_state_dicts_history()


