import os
import numpy as np
import torch
import random

# GLOBAL VARS
SINGLE_INPUT_SIZE = 18
N_LOOKAHEADS = 0
VALUE_INPUT_SIZE = SINGLE_INPUT_SIZE * (N_LOOKAHEADS + 1)
POLICY_INPUT_SIZE = VALUE_INPUT_SIZE + 4
BOARD_SIZE = 21
ACTION_SIZE = 14


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_pytorch_for_mpi(num_procs):
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    pid = os.getpid()
    # print(f'Proc {pid}: Reporting original number of Torch threads as {torch.get_num_threads()}', flush=True)
    if torch.get_num_threads() == 1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs), 1)
    torch.set_num_threads(fair_num_threads)
    # print(f'Proc {pid}: Reporting new number of Torch threads as {torch.get_num_threads()}', flush=True)


def softmax(xs, axis):
    xs = xs - np.max(xs, axis=axis, keepdims=True)
    xs_exp = np.exp(xs)
    return xs_exp / xs_exp.sum(axis=axis, keepdims=True)