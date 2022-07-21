from omegaconf import OmegaConf

default = OmegaConf.create({
    'global_epochs': 1000000,  # one global epoch: collect many trajectories, gradient descent multiple epochs, test
    'num_cpus': 12,  # configure according to your device / n parallel processes when collecting trajectories
    'batch_size': 2**12,  # minibatch size for update
    'ppo_epochs': 10,  # update epochs in one global epoch
    'gamma': 1,  # discount factor
    'lam': 0.95,  # lambda for gae
    'seed': 0,  # seed
    'test_every': 100,  # test every n global epochs / divisor of 100
    'save_every': 1000,  # save every n global epochs
    'pretrained_path': 'best_imitate.pth',  # will resume training from this checkpoint if specified
    'load_optimizer': True,  # if pretrained_path is specified, whether to load optimizer state_dict
    'anneal_lr': False,  # decrease learning rate linearly
    'buffer_size': int(1e5),  # ppo buffer max size to save data points
    'hidden_size': 32,  # hidden size of neural network
    'n_blocks': 2,  # n residual blocks for policy network
    'lr': 1e-3,  # learning rate to perform gradient descent
    'clip_coef': 0.3,  # clip coefficient of ppo
    'max_grad_norm': 0.5,  # perform gradient clip with this norm
    'vf_coef': 1,  # multiplied to value loss
    'ent_coef': 0.001,  # multiplied to entropy loss
    'fixed_opponent': 'beta',  # '' or among [random_agent, balanced_agent, beta_agent] / if set, always play against fixed opponent
    'against_prev_prob': 0.2,  # probability to play with previous self instead of current self
    'len_state_dicts_queue': 1000,  # max length of state dicts queue
    'state_dicts_priority_mul': 0.9,  # if certain historical agent loses against current agent then multiplied, if wins then divided to its priority
    'append_state_dict_interval': 10,  # append state dict to history every n global epochs
    'lam1_ge': 10,  # set lam to 1 for n epochs for initially stable value function learning
    'only_value_ge': 20,  # only learn value function initially for n epochs
    'warmup_epochs': 0,  # warmup learning rate for n global epochs
    'dense_reward': False,  # whether to use dense reward in environment (sparse reward: [-1, 1] at the end)
    'reset_return_stats': True,  # whether to reset return mean variance (True when moving to sparse reward from dense reward)
    'reset_policy_value_stats': False,
    'against_beta_prob': 0
})

resume = OmegaConf.create({
    'global_epochs': 1000000,  # one global epoch: collect many trajectories, gradient descent multiple epochs, test
    'num_cpus': 12,  # configure according to your device / n parallel processes when collecting trajectories
    'batch_size': 2**12,  # minibatch size for update
    'ppo_epochs': 10,  # update epochs in one global epoch
    'gamma': 1,  # discount factor
    'lam': 0.95,  # lambda for gae
    'seed': 0,  # seed
    'test_every': 100,  # test every n global epochs / divisor of 100
    'save_every': 1000,  # save every n global epochs
    'pretrained_path': 'pretrained_agents/0711-3007e.pth',  # 'pretrained_agents/0704-1501e.pth',  # will resume training from this checkpoint if specified
    'load_optimizer': True,  # if pretrained_path is specified, whether to load optimizer state_dict
    'anneal_lr': False,  # decrease learning rate linearly
    'buffer_size': int(1e5),  # ppo buffer max size to save data points
    'hidden_size': 32,  # hidden size of neural network
    'n_blocks': 2,  # n residual blocks for policy network
    'lr': 1e-4,  # learning rate to perform gradient descent
    'clip_coef': 0.3,  # clip coefficient of ppo
    'max_grad_norm': 0.5,  # perform gradient clip with this norm
    'vf_coef': 0.1,  # multiplied to value loss
    'ent_coef': 0.0001,  # multiplied to entropy loss
    'fixed_opponent': 'beta',  # '' or among [random, balanced, beta] / if set, always play against fixed opponent
    'against_prev_prob': 0.2,  # probability to play with previous self instead of current self
    'len_state_dicts_queue': 1000,  # max length of state dicts queue
    'state_dicts_priority_mul': 0.9,  # if certain historical agent loses against current agent then multiplied, if wins then divided to its priority
    'append_state_dict_interval': 10,  # append state dict to history every n global epochs
    'lam1_ge': 10,  # set lam to 1 for n epochs for initially stable value function learning
    'only_value_ge': 0,  # only learn value function initially for n epochs
    'warmup_epochs': 0,  # warmup learning rate for n global epochs
    'dense_reward': False,  # whether to use dense reward in environment (sparse reward: [-1, 1] at the end)
    'reset_return_stats': False,  # whether to reset return mean variance (True when moving to sparse reward from (dense reward / imitation))
    'reset_policy_value_stats': False,
    'against_beta_prob': 0.5
})

imitate = OmegaConf.create({
    'batch_size': 2**12,  # minibatch size for update  2**12
    'epochs': 100,  # training epochs
    'seed': 0,  # seed
    'pretrained_path': '',  # will resume training from this checkpoint if specified
    'hidden_size': 32,  # hidden size of neural network
    'n_blocks': 2,  # n residual blocks for policy network
    'lr': 1e-2,  # learning rate to perform gradient descent
    'num_workers': 8,  # num workers for pytorch dataloader
    'num_cpus': 12,  # configure according to your device / n parallel processes when collecting trajectories
})


configs = {
    'default': default,
    'resume': resume,
    'imitate': imitate
}
