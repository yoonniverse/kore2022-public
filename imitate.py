import wandb
from omegaconf import OmegaConf
from argparse import ArgumentParser
import copy
from tqdm import tqdm
import numpy as np
import joblib
import ray
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from multiprocessing import Pool
from collections import deque

from env import has_won
from agent import Agent
from configs import configs
from utils import *
from test import balanced_agent, beta_agent, run_test_remote


class KoreDataset(Dataset):

    def __init__(self):
        path = Path('data/processed')
        self.paths = list(path.glob('*.jl'))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return joblib.load(self.paths[idx])


def get_policy_inputs_mean_var(paths):
    res = np.zeros((len(paths), POLICY_INPUT_SIZE, BOARD_SIZE, BOARD_SIZE))
    for i, path in enumerate(paths):
        res[i] = joblib.load(path)['policy_inputs']
    mean = res.mean(axis=(0,2,3))
    var = res.var(axis=(0,2,3))
    return mean, var


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--run_name', type=str, default='imitate')
    parser.add_argument('--config_name', type=str, default='imitate')
    args = parser.parse_args()

    cfg = configs[args.config_name]
    print(cfg)
    seed_all(cfg.seed)

    wandb.init(
        project='kore2022',
        config=OmegaConf.to_container(cfg),
        name=f'imitate/{args.run_name}',
        save_code=True,
    )
    
    agent = Agent(copy.deepcopy(cfg))
    print('# parameters:', sum(p.numel() for p in agent.parameters() if p.requires_grad))

    if cfg.pretrained_path:
        ckp = torch.load(cfg.pretrained_path, map_location='cpu')
        agent.load_state_dict(ckp['state_dict'])
        agent.optimizer.load_state_dict(ckp['optimizer_state_dict'])
        print('loaded pretrained weights from', cfg.pretrained_path)
        del ckp

    dataset = KoreDataset()
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    state_dicts_queue = deque([copy.deepcopy(agent.state_dict())], maxlen=1000)

    # set policy state mean var
    path_chunks = np.array_split(dataset.paths, len(dataset.paths)//1e3)
    with Pool(cfg.num_cpus) as pool:
        out = list(tqdm(pool.imap(get_policy_inputs_mean_var, path_chunks), total=len(path_chunks)))
    tmp = agent.actor_critic.policy_state_mean_var
    tmp.mean.data = torch.tensor(np.mean([x[0] for x in out], axis=0), dtype=tmp.mean.dtype, device=tmp.mean.device)
    tmp.var.data = torch.tensor(np.mean([x[1] for x in out], axis=0), dtype=tmp.var.dtype, device=tmp.var.device)
    tmp.count.data = torch.tensor(len(dataset.paths), dtype=tmp.count.dtype, device=tmp.count.device)
    print('updated policy input stats to', agent.actor_critic.policy_state_mean_var.mean, agent.actor_critic.policy_state_mean_var.var, agent.actor_critic.policy_state_mean_var.count)

    # train
    best_score = 0
    for epoch in range(cfg.epochs):
        print(f'EPOCH {epoch}')
        wandb_logs = {'loss': 0, 'accuracy': 0}
        agent.set_device('cuda:0')
        losses = []
        accuracies = []
        agent.train()
        for data in tqdm(dataloader):
            loss, accuracy = agent.learn_imitation(data['policy_inputs'], data['mask_inputs'], data['targets'])
            losses.append(loss)
            accuracies.append(accuracy)
        agent.eval()

        wandb_logs['loss'] = np.mean(losses)
        wandb_logs['accuracy'] = np.mean(accuracies)
        if (epoch+1) % 1 == 0:
            agent.set_device('cpu')
            ckp = {'cfg': agent.cfg, 'state_dict': agent.state_dict()}
            against_balanced_envs = ray.get([run_test_remote.remote(ckp, balanced_agent, i, 1 if i < cfg.num_cpus else 0) for i in range(cfg.num_cpus * 2)])
            winrate_against_balanced = np.mean([has_won(e) if i < cfg.num_cpus else 1 - has_won(e) for i, e in enumerate(against_balanced_envs)])
            if winrate_against_balanced >= best_score:
                best_score = winrate_against_balanced
                agent.save('best_imitate.pth')
            wandb_logs['winrate_against_balanced'] = winrate_against_balanced
            # against_beta_envs = ray.get([run_test_remote.remote(ckp, beta_agent, i, 1 if i < cfg.num_cpus else 0) for i in range(cfg.num_cpus * 2)])
            # winrate_against_beta = np.mean([has_won(e) if i < cfg.num_cpus else 1 - has_won(e) for i, e in enumerate(against_beta_envs)])
            # wandb_logs['winrate_against_beta'] = winrate_against_beta
            # prev_agent = Agent(cfg)
            # prev_agent.load_state_dict(state_dicts_queue[-1])
            # against_prev_envs = ray.get([run_test_remote.remote(ckp, prev_agent.act, i, 1 if i < cfg.num_cpus else 0) for i in range(cfg.num_cpus * 2)])
            # winrate_against_prev = np.mean([has_won(e) if i < cfg.num_cpus else 1 - has_won(e) for i, e in enumerate(against_prev_envs)])
            # wandb_logs['winrate_against_prev'] = winrate_against_prev
        wandb.log(wandb_logs)
        print(wandb_logs)
        agent.save('imitate.pth')

        state_dicts_queue.append(copy.deepcopy(agent.state_dict()))


