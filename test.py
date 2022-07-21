from kaggle_environments import make
import ray
from tqdm import tqdm
from argparse import ArgumentParser
from kaggle_environments.envs.kore_fleets.kore_fleets import balanced_agent

from beta_agent import agent as beta_agent
from agent import Agent
from processors import process_obs, reverse_obs, process_action
from env import has_won
from utils import *
from configs import configs

random_agent = Agent(configs['default']).act


def run_test(ckp, opponent, seed, agent_idx):
    seed_all(seed)
    env = make("kore_fleets")

    agent = Agent(ckp['cfg'])
    agent.load_state_dict(ckp['state_dict'])
    if agent_idx == 0:
        _ = env.run([agent.act, opponent])
    else:
        _ = env.run([opponent, agent.act])
    if (env.steps[-1][0]['reward'] is None) or (env.steps[-1][1]['reward'] is None):
        print('RETRY TEST!')
        return run_test(ckp, opponent, seed, agent_idx)
    return env


@ray.remote
def run_test_remote(ckp, opponent, seed, agent_idx):
    return run_test(ckp, opponent, seed, agent_idx)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--num_cpus', type=int, default=14)
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--opponent', type=str, default='beta')
    parser.add_argument('--ckp_path', type=str, default='agent.pth')
    args = parser.parse_args()

    opponent_dict = {
        'random': random_agent,
        'balanced': balanced_agent,
        'beta': beta_agent
    }
    ckp = torch.load(args.ckp_path, map_location='cpu')
    if args.num_cpus > 1:
        ray.init(num_cpus=args.num_cpus)
        envs = ray.get([run_test_remote.remote(ckp, opponent_dict[args.opponent], i, 0 if i<args.n_episodes//2 else 1) for i in range(args.n_episodes)])
    else:
        envs = [run_test(ckp, opponent_dict[args.opponent], i, 0 if i<args.n_episodes//2 else 1) for i in tqdm(range(args.n_episodes))]
    wins = [has_won(e) if i<args.n_episodes//2 else 1-has_won(e) for i, e in enumerate(envs)]
    print("Win rate against baseline:", np.mean(wins))
