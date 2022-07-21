import shutil
from pathlib import Path
import json
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import joblib

from processors import process_obs, deprocess_action
from utils import BOARD_SIZE


def process_one(path):
    with open(path, 'r') as f:
        match = json.load(f)
    if 'ERROR' in match['statuses']:
        return
    try:
        player = np.argmax(match['rewards'])
    except TypeError as e:
        print(e)
        return
    policy_inputs = []
    mask_inputs = []
    targets = []
    for step in range(len(match['steps'])-1):
        obs = match['steps'][step][0]['observation']
        obs['player'] = player
        if len(obs['players'][player][1]) == 0:
            continue
        action = match['steps'][step+1][player]['action']
        raw_action = deprocess_action(action, obs, match['configuration'])
        if (raw_action == -1).all():
            continue
        policy_input, value_input, mask_input = process_obs(obs, match['configuration'])
        # for i, (p, m, t) in enumerate(zip(policy_input, mask_input, raw_action)):
        #     type, position = np.divmod(t, BOARD_SIZE ** 2)
        #     h, w = np.divmod(position, BOARD_SIZE)
        #     if not m[int(type), int(h), int(w)]:
        #         print('WHAT?>')
        #         joblib.dump((action, obs, i, p, m, t), 'tmp.jl')
        assert len(policy_input) == len(raw_action)
        policy_inputs.append(policy_input)
        mask_inputs.append(mask_input)
        targets.append(raw_action)
    policy_inputs, mask_inputs, targets = np.concatenate(policy_inputs, 0), np.concatenate(mask_inputs, 0), np.concatenate(targets, 0)
    rel_idx = targets != -1
    for i, (p, m, t) in enumerate(zip(policy_inputs, mask_inputs, targets)):
        type, position = np.divmod(t, BOARD_SIZE**2)
        h, w = np.divmod(position, BOARD_SIZE)
        if not m[int(type), int(h), int(w)]:
            rel_idx[i] = False
    name = path.split('/')[-1][:-5]
    rel_idx = np.where(rel_idx)[0]
    for i in range(len(rel_idx)):
        joblib.dump({'policy_inputs': policy_inputs[rel_idx[i]], 'mask_inputs': mask_inputs[rel_idx[i]], 'targets': targets[rel_idx[i]]}, f'data/processed/{name}_{i}.jl')


if __name__ == '__main__':

    num_cpus = 14
    shutil.rmtree('data/processed')
    Path('data/processed').mkdir(exist_ok=True)
    paths = [str(x) for x in Path('data/raw').glob('*.json') if not str(x).endswith('_info.json')]
    with Pool(num_cpus) as pool:
        out = list(tqdm(pool.imap(process_one, paths), total=len(paths)))