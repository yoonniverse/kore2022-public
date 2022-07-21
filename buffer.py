import numpy as np
from collections import Counter
from utils import *

# gae calculation

# 1. spinning up version
# def discount_cumsum(x, discount):
#     return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
#
#
# def cal_adv_ret(reward, value, gamma, lam):
#     last_val = 0
#     rews = np.append(reward, last_val)
#     vals = np.append(value, last_val)
#     deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
#     adv = discount_cumsum(deltas, gamma * lam)
#     ret = discount_cumsum(rews, gamma)[:-1]
#     return adv, ret


# 2. https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737
# (assuming last step of every trajectory is done)
def cal_adv_ret(rewards, values, gamma, lam):
    n_steps = len(rewards)
    values = np.append(values, 0)
    advantages = np.zeros(n_steps+1, dtype=np.float32)
    for t in reversed(range(n_steps)):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        advantages[t] = delta + gamma * lam * advantages[t+1]
    advantages = advantages[:n_steps]
    values = values[:n_steps]
    returns = advantages + values
    return advantages, returns


class PPOBuffer:

    def __init__(self, buffer_size):
        """
        Fixed-size buffer to store experience tuples.
        :param buffer_size: (int)
        """
        self.memory = {
            'policy_state': np.zeros((buffer_size, POLICY_INPUT_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            'value_state': np.zeros((buffer_size, VALUE_INPUT_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            'action': np.zeros(buffer_size, dtype=np.float32),
            'log_prob': np.zeros(buffer_size, dtype=np.float32),
            'return': np.zeros(buffer_size, dtype=np.float32),
            'advantage': np.zeros(buffer_size, dtype=np.float32),
            'action_mask': np.zeros((buffer_size, ACTION_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
        }
        self.memory_keys = set(self.memory.keys())
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0

    def add(self, experience_dict):
        """
        Add a new experience to memory.
        :param experience_dict: experience dictionary with keys {state, action, log_prob, return, advantage}
        each with shape (batch_size, *)
        """
        len_e = len(experience_dict[list(experience_dict.keys())[0]])
        assert self.memory_keys == set(experience_dict.keys())
        for k in self.memory_keys:
            len_cur_e = len(experience_dict[k])
            assert len_e == len_cur_e
            self.memory[k][self.ptr:self.ptr+len_cur_e] = experience_dict[k]
        self.ptr = (self.ptr+len_e) % self.buffer_size
        self.size = min(self.size+len_e, self.buffer_size)

    def update_from_trajectory(self, trajectory, gamma, lam):
        for k, v in trajectory.items():
            trajectory[k] = np.stack(v).astype(np.float32)
        rel_idx = trajectory['step'].astype(int)
        # mapper = dict(Counter(rel_idx))
        # trajectory['reward'] /= np.array([mapper[x] if x in mapper.keys() else 1 for x in range(len(trajectory['reward']))])  # divide reward evenly to each shipyard
        adv, ret = cal_adv_ret(trajectory['reward'], trajectory['value'], gamma, lam)
        e = {
            'policy_state': trajectory['policy_state'],
            'value_state': trajectory['value_state'][rel_idx],
            'action': trajectory['action'],
            'log_prob': trajectory['log_prob'],
            'return': ret[rel_idx],
            'advantage': adv[rel_idx],  # / np.array([mapper[x] for x in rel_idx]), equally split advantage to shipyards
            'action_mask': trajectory['action_mask']
        }
        self.add(e)
        return {**e, 'reward': trajectory['reward'], 'value': trajectory['value'], 'step': trajectory['step'], 'logit_map': trajectory['logit_map']}

    def reset(self):
        self.ptr = 0
        self.size = 0

    def sample(self, idx):
        out = {k: torch.from_numpy(self.memory[k][idx]) for k in self.memory_keys}
        out['advantage'] = (out['advantage'] - out['advantage'].mean()) / (out['advantage'].std() + 1e-8)
        return out

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return self.size