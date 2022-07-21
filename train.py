import copy
import gc
import time

import joblib
from argparse import ArgumentParser
from tqdm import tqdm
import ray
import wandb
from omegaconf import OmegaConf
from collections import deque

from env import Kore2022, get_stats_from_env, has_won, get_assets_from_env
from agent import Agent
from buffer import PPOBuffer
from configs import configs
from test import random_agent, balanced_agent, beta_agent, run_test_remote
from utils import *


def collect_episode(cfg, agent_state_dicts, seed, collect_idx):
    setup_pytorch_for_mpi(cfg.num_cpus)
    seed_all(seed)
    env = Kore2022(dense_reward=cfg.dense_reward)
    env_config = env.simulator.configuration
    agents = [Agent(cfg) for _ in range(len(agent_state_dicts))]
    for i, sd in enumerate(agent_state_dicts):
        agents[i].load_state_dict(sd)
    obs, _, _ = env.reset()
    done = False
    while not done:
        total_action = []
        for i in range(len(agents)):
            obs['player'] = i
            action = agents[i].act(obs, env_config)
            total_action.append(action)
        obs, reward, done = env.step(total_action)
        for i in range(len(agents)):
            agents[i].trajectory['reward'].append(reward[i])
            agents[i].trajectory['done'].append(done)
    trajectories = []
    for i in collect_idx:
        trajectories.append(agents[i].trajectory)
    return env, trajectories


def collect_episode_against_fixed(cfg, agent_state_dict, opponent, seed, agent_idx):
    setup_pytorch_for_mpi(cfg.num_cpus)
    seed_all(seed)
    env = Kore2022(dense_reward=cfg.dense_reward)
    env_config = env.simulator.configuration
    agent = Agent(cfg)
    agent.load_state_dict(agent_state_dict)
    obs, _, _ = env.reset()
    done = False
    while not done:
        obs['player'] = agent_idx
        processed_action0 = agent.act(obs, env_config)
        obs['player'] = 1 - agent_idx
        processed_action1 = opponent(obs, env_config)
        actions = [processed_action0, processed_action1] if agent_idx == 0 else [processed_action1, processed_action0]
        obs, reward, done = env.step(actions)
        agent.trajectory['reward'].append(reward[agent_idx])
        agent.trajectory['done'].append(done)
    trajectories = [agent.trajectory]
    gc.collect()
    return env, trajectories


@ray.remote
def collect_episode_remote(cfg, agent_state_dicts, seed, collect_idx):
    return collect_episode(cfg, agent_state_dicts, seed, collect_idx)


@ray.remote
def collect_episode_against_fixed_remote(cfg, agent_state_dict, opponent, seed, agent_idx):
    return collect_episode_against_fixed(cfg, agent_state_dict, opponent, seed, agent_idx)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--run_name', type=str, default='default')
    parser.add_argument('--config_name', type=str, default='default')
    args = parser.parse_args()

    cfg = configs[args.config_name]
    print(cfg)
    seed_all(cfg.seed)

    ray.init(num_cpus=cfg.num_cpus)
    # setup_pytorch_for_mpi(cfg.num_cpus)

    wandb.init(
        project='kore2022',
        config=OmegaConf.to_container(cfg),
        name=args.run_name,
        save_code=True,
    )

    agent = Agent(copy.deepcopy(cfg))
    print('# parameters:', sum(p.numel() for p in agent.parameters() if p.requires_grad))
    start_ge = 0
    if cfg.pretrained_path:
        agent.load(cfg.pretrained_path, cfg.load_optimizer)
        start_ge = agent.ge + 1
        print('loaded pretrained weights from', cfg.pretrained_path)

    buffer = PPOBuffer(buffer_size=cfg.buffer_size)

    against_beta = False
    pbar = tqdm(range(cfg.global_epochs+1))
    for ge in pbar:
        if ge < start_ge:
            continue
        t0 = time.time()

        wandb_logs = {'global_epoch': ge, 'actor_loss': 0, 'critic_loss': 0, 'entropy_loss': 0, 'approx_kl': 0,
                      'clip_frac': 0, 'loss_decrease': 0, 'kores_mean': 0, 'kores_var': 1, 'buffer_size': 0,
                      'mean_last_kores': 0, 'mean_last_asset': 0, 'mean_last_fleets': 0, 'mean_nspawns': 0, 'mean_nlaunches': 0, 'mean_nbuilds': 0,
                      'mean_last_board_kores': 0, 'max_abs_grad': 0, 'lr': 0}

        # reset
        buffer.reset()
        agent.reset_trajectory()

        # sort state dict
        # agent.sort_state_dicts_history()

        # collect data
        fixed_opponent_dict = {'random': random_agent, 'beta': beta_agent, 'balanced': balanced_agent}
        out = []
        run_num_to_queue_idx = {}
        collect_idx_lst = []
        for i in range(cfg.num_cpus):
            cur_seed = cfg.seed + ge * cfg.num_cpus + i
            if cfg.fixed_opponent:
                out.append(collect_episode_against_fixed_remote.remote(cfg, agent.state_dict(), fixed_opponent_dict[cfg.fixed_opponent], cur_seed, 0 if i < cfg.num_cpus // 2 else 1))
                collect_idx_lst.append([0])
            elif np.random.rand() < cfg.against_beta_prob:
                out.append(collect_episode_against_fixed_remote.remote(cfg, agent.state_dict(), beta_agent, cur_seed, 0 if i < cfg.num_cpus // 2 else 1))
                collect_idx_lst.append([0])
            else:
                state_dicts_to_pass = [agent.state_dict(), agent.state_dict()]
                collect_idx = [0, 1]
                if (len(agent.state_dicts_history) > 10) and (np.random.rand() < cfg.against_prev_prob):
                    clipped_priority = np.clip(agent.state_dicts_priority, -10, 10)
                    sample_dist = clipped_priority / clipped_priority.sum()
                    queue_idx = np.random.choice(len(agent.state_dicts_history), p=sample_dist)
                    run_num_to_queue_idx[i] = queue_idx
                    if i < cfg.num_cpus // 2:
                        state_dicts_to_pass[1] = agent.state_dicts_history[queue_idx]
                        collect_idx = [0]
                    else:
                        state_dicts_to_pass[0] = agent.state_dicts_history[queue_idx]
                        collect_idx = [1]
                out.append(collect_episode_remote.remote(cfg, state_dicts_to_pass, cur_seed, collect_idx))
                collect_idx_lst.append(collect_idx)
        out = ray.get(out)

        # update state dicts priority
        if (not cfg.fixed_opponent) and (cfg.against_prev_prob > 0):
            for k, v in run_num_to_queue_idx.items():
                won = has_won(out[k][0].simulator)
                if k >= cfg.num_cpus // 2: won = 1-won
                if won:
                    agent.state_dicts_priority[v] *= cfg.state_dicts_priority_mul
                else:
                    agent.state_dicts_priority[v] /= cfg.state_dicts_priority_mul

        trajectory_envs = []
        trajectories = []
        for env_, trajectories_ in out:
            trajectory_envs.append(env_)
            for trajectory_ in trajectories_:
                trajectory = buffer.update_from_trajectory(trajectory_, cfg.gamma, 1 if ge <= cfg.lam1_ge else cfg.lam)
                trajectories.append(trajectory)
        joblib.dump(trajectory_envs[0], 'trajectory_env.jl')
        joblib.dump(trajectories[0], 'trajectory.jl')

        # log board statistics
        board_statistics = {'mean_nsteps': [], 'mean_last_kores': [], 'mean_last_asset': [], 'mean_last_shipyards': [],
                            'mean_last_fleets': [], 'mean_nspawns': [], 'mean_nlaunches': [], 'mean_sum_launch_ships': [],
                            'mean_nbuilds': [], 'mean_last_board_kores': []}
        for env_, collect_idx_ in zip(trajectory_envs, collect_idx_lst):
            stats = get_stats_from_env(env_.simulator)
            for i_ in collect_idx_:
                for k, v in stats[i_].items():
                    board_statistics[f'mean_{k}'].append(v)
        for k, v in board_statistics.items():
            wandb_logs[k] = np.mean(v)

        if ge == start_ge:
            # first, initialize normalization coefficients
            if cfg.reset_policy_value_stats:
                agent.actor_critic.policy_state_mean_var.reset()
                agent.actor_critic.policy_state_mean_var.update(torch.from_numpy(buffer.memory['policy_state'][:len(buffer)]).to(device=agent.device), dim=(0,2,3))
                agent.actor_critic.value_state_mean_var.update(torch.from_numpy(buffer.memory['value_state'][:len(buffer)]).to(device=agent.device), dim=(0,2,3))
                print(agent.actor_critic.policy_state_mean_var.mean, agent.actor_critic.policy_state_mean_var.var)
                print(agent.actor_critic.value_state_mean_var.mean, agent.actor_critic.value_state_mean_var.var)
                continue
            if cfg.reset_return_stats:
                agent.actor_critic.return_mean_var.reset()
                agent.actor_critic.return_mean_var.update(torch.from_numpy(buffer.memory['return'][:len(buffer)]).to(device=agent.device), dim=0)
                print(agent.actor_critic.return_mean_var.mean, agent.actor_critic.return_mean_var.var)
                continue

        # adjust learning rate
        assert ge >= 1
        cur_lr = cfg.lr
        if ge <= cfg.warmup_epochs:
            cur_lr = 1e-7 + (cfg.lr - 1e-7) * (ge - 1) / cfg.warmup_epochs
            for group in agent.optimizer.param_groups:
                group['lr'] = cur_lr
        elif cfg.anneal_lr:
            cur_lr = agent.lr * (1 - (ge - 1) / cfg.global_epochs)
            for group in agent.optimizer.param_groups:
                group['lr'] = cur_lr
        wandb_logs['lr'] = cur_lr

        # train
        agent.set_device('cuda:0')
        agent.train()
        losses = []
        for ppo_epoch in range(cfg.ppo_epochs):
            batch_index = np.arange(len(buffer))
            np.random.shuffle(batch_index)
            loss = 0
            batch_size = min(cfg.batch_size, len(buffer))
            n_minibatches = np.ceil(len(buffer)/batch_size)
            for i in range(0, len(buffer) - batch_size + 1, batch_size):
                minibatch_index = batch_index[i:i + batch_size]
                e = buffer.sample(minibatch_index)
                cur_loss, cur_logs = agent.learn(e)
                loss += cur_loss / n_minibatches
                if ppo_epoch == cfg.ppo_epochs-1:
                    for k, v in cur_logs.items():
                        wandb_logs[k] += v / n_minibatches
            losses.append(loss)

        wandb_logs['buffer_size'] = len(buffer)
        wandb_logs['loss_decrease'] = losses[0] - losses[-1]
        agent.set_device('cpu')
        agent.eval()

        # test
        if (ge % cfg.test_every == 0) or ((ge < 100) and (ge % 20 == 0)):
            # test against prev
            for prev in [100, 300, 1000]:
                queue_prev = int(prev / cfg.append_state_dict_interval)
                if len(agent.state_dicts_history) >= queue_prev:
                    out0 = ray.get([collect_episode_remote.remote(cfg, [agent.state_dict(), agent.state_dicts_history[-queue_prev]], i, [0]) for i in range(cfg.num_cpus)])
                    winrate0 = np.mean([has_won(e.simulator) for e, _ in out0])
                    asset_diff0 = np.mean([e.asset_diff for e, _ in out0])
                    out1 = ray.get([collect_episode_remote.remote(cfg, [agent.state_dicts_history[-queue_prev], agent.state_dict()], i+cfg.num_cpus, [1]) for i in range(cfg.num_cpus)])
                    winrate1 = np.mean([(1-has_won(e.simulator)) for e, _ in out1])
                    asset_diff1 = np.mean([-e.asset_diff for e, _ in out1])
                    wandb_logs[f'winrate_against_prev{prev}'] = (winrate0 + winrate1) / 2
                    wandb_logs[f'asset_diff_against_prev{prev}'] = (asset_diff0 + asset_diff1) / 2
            # test against fixed
            ckp = {'cfg': agent.cfg, 'state_dict': agent.state_dict()}
            # against_balanced_envs = ray.get([run_test_remote.remote(ckp, balanced_agent, i, 0 if i < cfg.num_cpus else 1) for i in range(cfg.num_cpus*2)])
            # wandb_logs['winrate_against_balanced'] = np.mean([has_won(e) if i < cfg.num_cpus else 1-has_won(e) for i, e in enumerate(against_balanced_envs)])
            # wandb_logs['asset_diff_against_balanced'] = np.mean([-np.diff(get_assets_from_env(e)) if i < cfg.num_cpus else np.diff(get_assets_from_env(e)) for i, e in enumerate(against_balanced_envs)])
            # if wandb_logs['winrate_against_balanced'] > 0.8:  # if our agent defeats balanced, start to test against beta
            #     against_beta = True
            # if against_beta:
            against_beta_envs = ray.get([run_test_remote.remote(ckp, beta_agent, i, 0 if i < cfg.num_cpus else 1) for i in range(cfg.num_cpus*2)])
            wandb_logs['winrate_against_beta'] = np.mean([has_won(e) if i < cfg.num_cpus else 1-has_won(e) for i, e in enumerate(against_beta_envs)])
            wandb_logs['asset_diff_against_beta'] = np.mean([-np.diff(get_assets_from_env(e)) if i < cfg.num_cpus else np.diff(get_assets_from_env(e)) for i, e in enumerate(against_beta_envs)])
            # else:
            #     against_random_envs = ray.get([run_test_remote.remote(ckp, random_agent, i, 0 if i < cfg.num_cpus else 1) for i in range(cfg.num_cpus*2)])
            #     wandb_logs['winrate_against_random'] = np.mean([has_won(e) if i < cfg.num_cpus else 1-has_won(e) for i, e in enumerate(against_random_envs)])
            #     wandb_logs['asset_diff_against_random'] = np.mean([-np.diff(get_assets_from_env(e)) if i < cfg.num_cpus else np.diff(get_assets_from_env(e)) for i, e in enumerate(against_random_envs)])


        # update network state mean, var
        agent.actor_critic.policy_state_mean_var.update(torch.from_numpy(buffer.memory['policy_state'][:len(buffer)]).to(device=agent.device), dim=(0,2,3))
        agent.actor_critic.value_state_mean_var.update(torch.from_numpy(buffer.memory['value_state'][:len(buffer)]).to(device=agent.device), dim=(0,2,3))
        wandb_logs['kores_mean'] = agent.actor_critic.policy_state_mean_var.mean[0].item()
        wandb_logs['kores_var'] = agent.actor_critic.policy_state_mean_var.var[0].item()

        # update network return mean, var
        agent.actor_critic.return_mean_var.update(torch.from_numpy(buffer.memory['return'][:len(buffer)]).to(device=agent.device), dim=0)
        wandb_logs['return_mean'] = agent.actor_critic.return_mean_var.mean[0].item()
        wandb_logs['return_var'] = agent.actor_critic.return_mean_var.var[0].item()

        # update ge for agent
        agent.ge += 1

        # append state dicts
        if ge % cfg.append_state_dict_interval == 0:
            agent.append_state_dicts_history(agent.state_dict())

        # save
        agent.save('agent.pth')
        if ge % cfg.save_every == 0:
            agent.save(f'agent{ge}.pth')

        # log
        wandb_logs['runtime'] = time.time() - t0
        wandb.log(wandb_logs)
