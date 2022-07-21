from agent import Agent
import torch
import os
import omegaconf

path = '/kaggle_simulations/agent' if os.path.exists('/kaggle_simulations') else '.'
ckp = torch.load(f'{path}/agent.pth')
agent = Agent(ckp['cfg'])
agent.load_state_dict(ckp['state_dict'])


def trained_agent(obs, config):
    return agent.act(obs, config)