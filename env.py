import numpy as np
import joblib
from kaggle_environments import make


def get_kores_from_env(env, step=-1):
    obs_players = env.steps[step][0]['observation']['players']
    return obs_players[0][0], obs_players[1][0]


def get_nshipyards_from_env(env, step=-1):
    obs_players = env.steps[step][0]['observation']['players']
    return len(obs_players[0][1]), len(obs_players[1][1])


def get_nfleets_from_env(env, step=-1):
    obs_players = env.steps[step][0]['observation']['players']
    return len(obs_players[0][2]), len(obs_players[1][2])


def get_asset_from_player(player):
    possessing_kores = player[0]
    shipyards = 50 * 10 * (len(player[1]))
    shipyard_ships = 10 * np.sum([x[1] for x in player[1].values()])
    launched_ships = 10 * np.sum([x[2] for x in player[2].values()])
    ship_kores = np.sum([x[1] for x in player[2].values()])
    return possessing_kores + shipyards + shipyard_ships + launched_ships + ship_kores


def get_assets_from_env(env, step=-1):
    obs_players = env.steps[step][0]['observation']['players']
    return get_asset_from_player(obs_players[0]), get_asset_from_player(obs_players[1])


def get_board_kores_from_env(env, step=-1):
    return np.sum(env.steps[step][0]['observation']['kore'])


def get_nspawns_nlaunches_nbuilds_from_action(action):
    stats = {'nspawns': 0, 'nlaunches': 0, 'sum_launch_ships': 0, 'nbuilds': 0}
    for a in action.values():
        if a.startswith('LAUNCH'):
            stats['nlaunches'] += 1
            stats['sum_launch_ships'] += int(a.split('_')[1])
            if 'C' in a.split('_')[-1]:
                stats['nbuilds'] += 1
        if a.startswith('SPAWN'):
            stats['nspawns'] += int(a.split('_')[1])
    return stats


def get_nspawns_nlaunches_nbuilds_from_env(env):
    p0_stats = {'nspawns': 0, 'nlaunches': 0, 'sum_launch_ships': 0, 'nbuilds': 0}
    p1_stats = {'nspawns': 0, 'nlaunches': 0, 'sum_launch_ships': 0, 'nbuilds': 0}
    for step in env.steps:
        p0_cur_stats = get_nspawns_nlaunches_nbuilds_from_action(step[0]['action'])
        for k, v in p0_cur_stats.items():
            p0_stats[k] += v
        p1_cur_stats = get_nspawns_nlaunches_nbuilds_from_action(step[1]['action'])
        for k, v in p1_cur_stats.items():
            p1_stats[k] += v
    return p0_stats, p1_stats


def get_stats_from_env(env):
    stats = get_nspawns_nlaunches_nbuilds_from_env(env)
    stats[0]['last_kores'], stats[1]['last_kores'] = get_kores_from_env(env)
    stats[0]['last_asset'], stats[1]['last_asset'] = get_assets_from_env(env)
    stats[0]['last_shipyards'], stats[1]['last_shipyards'] = get_nshipyards_from_env(env)
    stats[0]['last_fleets'], stats[1]['last_fleets'] = get_nfleets_from_env(env)
    stats[0]['nsteps'], stats[1]['nsteps'] = len(env.steps), len(env.steps)
    stats[0]['last_board_kores'] = stats[1]['last_board_kores'] = get_board_kores_from_env(env)
    return stats


def has_won(env, step=-1):
    return env.steps[step][0]['reward'] > env.steps[step][1]['reward']


class Kore2022:

    def __init__(self, dense_reward=False):
        self.simulator = make("kore_fleets", debug=True)
        self.asset_diff = 0.
        self.action_history = [[], []]
        self.dense_reward = dense_reward

    def step(self, actions):
        """
        obs: {'remainingOverageTime': int, 'step': int, 'player': 0 or 1, 'kore': list(441),
         'players': [[kore, {shipyardkey: [pos, n_ship, turns_controlled]}, {fleetkey: [pos, kore, n_ship, direction, remaining_plan]} for each player]}
        pos=(row * size + column)
        """
        out = self.simulator.step(actions)
        obs = out[0]['observation']
        asset0 = get_asset_from_player(obs['players'][0])
        asset1 = get_asset_from_player(obs['players'][1])
        asset_diff = asset0 - asset1
        if self.dense_reward:

            reward = asset_diff - self.asset_diff
            # reward /= (asset0 + asset1 + 1e-8)
            rewards = np.array([reward, -reward])
        else:  # sparse
            rewards = np.array([0, 0])
            if self.simulator.done:
                if out[0]['reward'] > out[1]['reward']:
                    rewards = np.array([1, -1])
                elif out[0]['reward'] < out[1]['reward']:
                    rewards = np.array([-1, 1])
        self.asset_diff = asset_diff

        return obs, rewards, self.simulator.done

    def reset(self):
        self.asset_diff = 0.
        self.action_history = [[], []]
        return self.simulator.reset(2)[0]['observation'], np.array([0, 0]), False