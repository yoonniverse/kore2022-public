from kaggle_environments.envs.kore_fleets.helpers import Board, Point, ShipyardAction, Fleet, Direction
import copy
from scipy.stats import rankdata

from utils import *
from env import get_asset_from_player

"""
index = i * 21 + j 
array(i, j): top-left(0,0) & bottom-right(20, 20)
Point(x, y): bottom-left(0,0) & top-right(20, 20)
"""

MOUNTAIN_ARRAY = np.array([[20., 19., 18., 17., 16., 15., 14., 13., 12., 11., 10., 11., 12.,
        13., 14., 15., 16., 17., 18., 19., 20.],
       [19., 18., 17., 16., 15., 14., 13., 12., 11., 10.,  9., 10., 11.,
        12., 13., 14., 15., 16., 17., 18., 19.],
       [18., 17., 16., 15., 14., 13., 12., 11., 10.,  9.,  8.,  9., 10.,
        11., 12., 13., 14., 15., 16., 17., 18.],
       [17., 16., 15., 14., 13., 12., 11., 10.,  9.,  8.,  7.,  8.,  9.,
        10., 11., 12., 13., 14., 15., 16., 17.],
       [16., 15., 14., 13., 12., 11., 10.,  9.,  8.,  7.,  6.,  7.,  8.,
         9., 10., 11., 12., 13., 14., 15., 16.],
       [15., 14., 13., 12., 11., 10.,  9.,  8.,  7.,  6.,  5.,  6.,  7.,
         8.,  9., 10., 11., 12., 13., 14., 15.],
       [14., 13., 12., 11., 10.,  9.,  8.,  7.,  6.,  5.,  4.,  5.,  6.,
         7.,  8.,  9., 10., 11., 12., 13., 14.],
       [13., 12., 11., 10.,  9.,  8.,  7.,  6.,  5.,  4.,  3.,  4.,  5.,
         6.,  7.,  8.,  9., 10., 11., 12., 13.],
       [12., 11., 10.,  9.,  8.,  7.,  6.,  5.,  4.,  3.,  2.,  3.,  4.,
         5.,  6.,  7.,  8.,  9., 10., 11., 12.],
       [11., 10.,  9.,  8.,  7.,  6.,  5.,  4.,  3.,  2.,  1.,  2.,  3.,
         4.,  5.,  6.,  7.,  8.,  9., 10., 11.],
       [10.,  9.,  8.,  7.,  6.,  5.,  4.,  3.,  2.,  1.,  0.,  1.,  2.,
         3.,  4.,  5.,  6.,  7.,  8.,  9., 10.],
       [11., 10.,  9.,  8.,  7.,  6.,  5.,  4.,  3.,  2.,  1.,  2.,  3.,
         4.,  5.,  6.,  7.,  8.,  9., 10., 11.],
       [12., 11., 10.,  9.,  8.,  7.,  6.,  5.,  4.,  3.,  2.,  3.,  4.,
         5.,  6.,  7.,  8.,  9., 10., 11., 12.],
       [13., 12., 11., 10.,  9.,  8.,  7.,  6.,  5.,  4.,  3.,  4.,  5.,
         6.,  7.,  8.,  9., 10., 11., 12., 13.],
       [14., 13., 12., 11., 10.,  9.,  8.,  7.,  6.,  5.,  4.,  5.,  6.,
         7.,  8.,  9., 10., 11., 12., 13., 14.],
       [15., 14., 13., 12., 11., 10.,  9.,  8.,  7.,  6.,  5.,  6.,  7.,
         8.,  9., 10., 11., 12., 13., 14., 15.],
       [16., 15., 14., 13., 12., 11., 10.,  9.,  8.,  7.,  6.,  7.,  8.,
         9., 10., 11., 12., 13., 14., 15., 16.],
       [17., 16., 15., 14., 13., 12., 11., 10.,  9.,  8.,  7.,  8.,  9.,
        10., 11., 12., 13., 14., 15., 16., 17.],
       [18., 17., 16., 15., 14., 13., 12., 11., 10.,  9.,  8.,  9., 10.,
        11., 12., 13., 14., 15., 16., 17., 18.],
       [19., 18., 17., 16., 15., 14., 13., 12., 11., 10.,  9., 10., 11.,
        12., 13., 14., 15., 16., 17., 18., 19.],
       [20., 19., 18., 17., 16., 15., 14., 13., 12., 11., 10., 11., 12.,
        13., 14., 15., 16., 17., 18., 19., 20.]])

middle = BOARD_SIZE//2
FILTERS = np.zeros((BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE))
for i in range(BOARD_SIZE):
    for j in range(BOARD_SIZE):
        if i < middle:
            FILTERS[i, j, i:middle + 1, [middle, j]] = 1
        elif i > middle:
            FILTERS[i, j, middle:i + 1, [middle, j]] = 1
        if j < middle:
            FILTERS[i, j, [middle, i], j:middle + 1] = 1
        elif j > middle:
            FILTERS[i, j, [middle, i], middle:j + 1] = 1
del middle


def decode_pos(pos):
    return pos // BOARD_SIZE, pos % BOARD_SIZE


def get_attack_feature(obs, env_config, max_len=30):
    board = Board(obs, env_config)
    def find_first_non_digit(candidate_str):
        for i in range(len(candidate_str)):
            if not candidate_str[i].isdigit():
                return i
        else:
            return len(candidate_str) + 1

    res = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    shipyards = list(board.shipyards.values())
    for shipyard in shipyards:
        location = Point(*shipyard.position).to_index(BOARD_SIZE)
        i, j = map(int, np.divmod(location, BOARD_SIZE))
        if shipyard.player_id == board.current_player.id:
            res[0, i, j] = shipyard.ship_count
        else:
            res[1, i, j] = shipyard.ship_count
    shipyard_locations = np.array([Point(*x.position).to_index(BOARD_SIZE) for x in shipyards])
    for fleet in board.fleets.values():
        direction = Direction(fleet.direction)
        location = Point(*fleet.position).to_index(BOARD_SIZE)
        remaining_plan = fleet.flight_plan
        counter = 0
        while counter < max_len:
            if location in shipyard_locations:
                shipyard = shipyards[np.argwhere(shipyard_locations == location).item()]
                if fleet.player_id != shipyard.player_id:
                    i, j = map(int, np.divmod(location, BOARD_SIZE))
                    if shipyard.player_id == board.current_player.id:
                        res[0, i, j] -= fleet.ship_count  # my shipyard will be attacked
                    else:
                        res[1, i, j] -= fleet.ship_count  # opponent shipyard will be attacked
                break
            if remaining_plan:
                while remaining_plan.startswith("0"):
                    remaining_plan = remaining_plan[1:]
                if remaining_plan[0] == "C":
                    break
                if remaining_plan[0].isalpha():
                    direction = Direction.from_char(remaining_plan[0])
                    remaining_plan = remaining_plan[1:]
                else:
                    idx = find_first_non_digit(remaining_plan)
                    digits = int(remaining_plan[:idx])
                    rest = remaining_plan[idx:]
                    digits -= 1
                    if digits > 0:
                        remaining_plan = str(digits) + rest
                    else:
                        remaining_plan = rest
            location = Point.from_index(location, BOARD_SIZE).translate(direction.to_point(), BOARD_SIZE).to_index(
                BOARD_SIZE)
            counter += 1

    return res


def process_obs_single_timestamp(obs, env_config):
    """
    obs
    {'remainingOverageTime': int, 'step': int, 'player': 0 or 1, 'kore': list(441),
     'players': [[kore, {shipyardkey: [pos, n_ship, turns_controlled]}, {fleetkey: [pos, kore, n_ship, direction, remaining_plan]} for each player]}
    pos=(row * size + column)

    res
    0 kore
    1 step
    2 0_shipyard
    3 0_shipyard_ship_count
    4 0_shipyard_turns_controlled
    5 0_fleet_kore
    6 0_fleet_ship_count
    7 0_broadcasted_possessing_kore
    8 0_broadcasted_possessing_asset
    9 1_shipyard
    10 1_shipyard_ship_count
    11 1_shipyard_turns_controlled
    12 1_fleet_kore
    13 1_fleet_ship_count
    14 1_broadcasted_possessing kore
    15 1_broadcasted_possessing_asset
    16 0_shipyard_ships_after_being_attacked
    17 1_shipyard_ships_after_being_attacked
    """
    # TODO: CHANGE <reverse_obs> & <utils.py STATE_SIZE> if you reengineer features
    res = np.zeros((SINGLE_INPUT_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    res[0] = np.reshape(obs['kore'], (BOARD_SIZE, BOARD_SIZE))
    res[1] = obs['step']

    player0_shipyard = list(obs['players'][0][1].values())
    for x in player0_shipyard:
        pos = decode_pos(x[0])
        res[2, pos[0], pos[1]] = 1
        res[3, pos[0], pos[1]] = x[1]
        res[4, pos[0], pos[1]] = x[2]
    player0_fleet = list(obs['players'][0][2].values())
    for x in player0_fleet:
        pos = decode_pos(x[0])
        res[5, pos[0], pos[1]] = x[1]
        res[6, pos[0], pos[1]] = x[2]
    res[7] = obs['players'][0][0]
    res[8] = get_asset_from_player(obs['players'][0])

    player1_shipyard = list(obs['players'][1][1].values())
    for x in player1_shipyard:
        pos = decode_pos(x[0])
        res[9, pos[0], pos[1]] = 1
        res[10, pos[0], pos[1]] = x[1]
        res[11, pos[0], pos[1]] = x[2]
    player1_fleet = list(obs['players'][1][2].values())
    for x in player1_fleet:
        pos = decode_pos(x[0])
        res[12, pos[0], pos[1]] = x[1]
        res[13, pos[0], pos[1]] = x[2]
    res[14] = obs['players'][1][0]
    res[15] = get_asset_from_player(obs['players'][1])
    # reverse obs if player == 1`
    if obs['player'] == 1:
        res = reverse_obs(res)
    res[[16, 17]] = get_attack_feature(obs, env_config)
    return res


def process_obs_base(obs, env_config, n_lookaheads=N_LOOKAHEADS):
    res = [process_obs_single_timestamp(obs, env_config)]
    board = Board(obs, env_config)
    for i in range(n_lookaheads):
        board = board.next()
        res.append(process_obs_single_timestamp(board.observation, env_config))
    res = np.concatenate(res, axis=0)
    return res


def get_expected_kores_mined_per_step(kores, shipyard_i, shipyard_j):
    mean_kores = np.zeros((BOARD_SIZE, BOARD_SIZE))
    middle = BOARD_SIZE // 2
    shifted_board = np.roll(kores, shift=(middle - shipyard_i, middle - shipyard_j), axis=(0, 1))
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            mean_kores[i, j] = (shifted_board * FILTERS[i, j]).sum() / (FILTERS[i, j].sum() - 1)
    mean_kores = np.roll(mean_kores, shift=(shipyard_i - middle, shipyard_j - middle), axis=(0, 1))
    return mean_kores


def calculate_asset_diff_single_cell(board, shipyard, target_position_index, build, horizontal_first, orig_asset_diffs):
    board.shipyards[shipyard.id].next_action = ShipyardAction.from_str(
        generate_launch_action(shipyard, Point.from_index(target_position_index, BOARD_SIZE), build=build, horizontal_first=horizontal_first, boomerang=False)
    )
    fleet_uid = f'{board.observation["step"] + 1}-1'
    player = board.observation['player']
    board = board.next()
    counter = 0
    while fleet_uid in board.observation['players'][player][2].keys():
        board = board.next()
        counter += 1
    asset_diff = get_asset_from_player(board.observation['players'][player]) - get_asset_from_player(board.observation['players'][1-player])
    return asset_diff - orig_asset_diffs[counter]


def process_obs(obs, env_config):
    res = []
    masks = []
    base = process_obs_base(obs, env_config)
    assert base.shape[0] == VALUE_INPUT_SIZE
    board = Board(obs, env_config)
    me = board.current_player
    shipyards = me.shipyards
    if len(shipyards) == 0:
        return None, base, None
    open_cells = np.ones((BOARD_SIZE, BOARD_SIZE))
    shipyard_cells = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.bool)
    for shipyard in board.shipyards.values():
        x, y = shipyard.position
        i, j = BOARD_SIZE - y - 1, x
        shipyard_cells[i, j] = 1
        cur_mountain_array = np.roll(MOUNTAIN_ARRAY, shift=(i - BOARD_SIZE // 2, j - BOARD_SIZE // 2), axis=(0, 1))
        open_cells *= cur_mountain_array >= 4

    for shipyard in shipyards:
        x, y = shipyard.position
        h, w = BOARD_SIZE - y - 1, x

        # generate input
        tmp = np.zeros((POLICY_INPUT_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        tmp[:VALUE_INPUT_SIZE] = base.copy()
        tmp[VALUE_INPUT_SIZE, h, w] = 1
        tmp[VALUE_INPUT_SIZE + 1] = tmp[3, h, w]
        tmp[VALUE_INPUT_SIZE + 2] = get_expected_kores_mined_per_step(tmp[0], h, w)
        tmp[VALUE_INPUT_SIZE + 3] = tmp[16, h, w]
        res.append(tmp)

        """
        generate action mask
        action
        0. spawn
        1. launch_min_horizontal
        2. launch_min_vertical
        3. launch_middle_horizontal
        4. launch_middle_vertical
        5. launch_max_horizontal
        6. launch_max_vertical
        7. build_min_horizontal
        8. build_min_vertical
        9. build_middle_horizontal
        10. build_middle_vertical
        11. build_max_horizontal
        12. build_max_vertical
        13. do_nothing
        rule
        Number Ships	Max Flight Plan Length
        1	1
        2	2
        3	3
        5	4
        8	5
        13	6
        21	7
        34	8
        55	9
        91	10
        149	11
        245	12
        404	13
        """
        mask = np.zeros((ACTION_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=np.bool)
        if shipyard.ship_count >= 2:
            mask[[1,2,3,4,5,6], (h+1)%BOARD_SIZE, w] = 1
            mask[[1,2,3,4,5,6], (h-1)%BOARD_SIZE, w] = 1
            mask[[1,2,3,4,5,6], h, (w+1)%BOARD_SIZE] = 1
            mask[[1,2,3,4,5,6], h, (w-1)%BOARD_SIZE] = 1
        if shipyard.ship_count >= 3:
            mask[[1,2,3,4,5,6], h, :] = 1
            mask[[1,2,3,4,5,6], :, w] = 1
        if shipyard.ship_count >= 5:
            mask[[1,2,3,4,5,6], (h+1)%BOARD_SIZE, (w+1)%BOARD_SIZE] = 1
            mask[[1,2,3,4,5,6], (h+1)%BOARD_SIZE, (w-1)%BOARD_SIZE] = 1
            mask[[1,2,3,4,5,6], (h-1)%BOARD_SIZE, (w+1)%BOARD_SIZE] = 1
            mask[[1,2,3,4,5,6], (h-1)%BOARD_SIZE, (w-1)%BOARD_SIZE] = 1
        if shipyard.ship_count >= 8:
            mask[[2,4,6], (h+1)%BOARD_SIZE, :] = 1
            mask[[2,4,6], (h-1)%BOARD_SIZE, :] = 1
            mask[[1,3,5], :, (w+1)%BOARD_SIZE] = 1
            mask[[1,3,5], :, (w-1)%BOARD_SIZE] = 1
        if shipyard.ship_count >= 13:
            mask[[1,2,3,4,5,6], (h+1)%BOARD_SIZE, :] = 1
            mask[[1,2,3,4,5,6], (h-1)%BOARD_SIZE, :] = 1
            mask[[1,2,3,4,5,6], :, (w+1)%BOARD_SIZE] = 1
            mask[[1,2,3,4,5,6], :, (w-1)%BOARD_SIZE] = 1
        if shipyard.ship_count >= 21:
            mask[:, :, :] = 1
        # build is zeroed when ship is less than 50
        if shipyard.ship_count < 50:
            mask[[7,8,9,10,11,12], :, :] = 0

        """
        heuristics
        """
        # build is invalid when ship is less than 100
        if shipyard.ship_count < 100:
            mask[[7,8,9,10,11,12], :, :] = 0
        # build is valid only where there is no kore and at open cells
        cur_mountain_array = np.roll(MOUNTAIN_ARRAY, shift=(h - BOARD_SIZE // 2, w - BOARD_SIZE // 2), axis=(0, 1))
        mask[[7,8,9,10,11,12], :, :] = mask[[7,8,9,10,11,12], :, :] * ((cur_mountain_array <= 8) * open_cells * (base[0, :, :] == 0))[None, :, :]
        # # launch and build is disabled if distance to target > 13
        # mask *= cur_mountain_array <= 13
        # launch min is invalid if distance >= 10
        mask[[1,2]] = mask[[1,2]] * (cur_mountain_array < 10)[None, :, :]
        # launch straight line is disabled
        nonstraight_cells = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.bool)
        nonstraight_cells[h, :] = 0
        nonstraight_cells[:, w] = 0
        mask[1:7] = mask[1:7] * (nonstraight_cells | shipyard_cells)[None, :, :]
        # if being attacked
        is_being_attacked = base[16, h, w] != base[3, h, w]
        mask[3:13, :, :] = mask[3:13, :, :] * ~is_being_attacked  # build is disabled, launch middle/max is disabled
        mask[1:3] = mask[1:3] * ~(is_being_attacked & (base[16, h, w] < 21))  # launch min is disabled if resulting ships < 21
        # launch to one of most 10 profitable locations or to shipyard
        for i_ in range(1, 7):
            mask[i_] *= ((rankdata(-tmp[VALUE_INPUT_SIZE + 2]*mask[i_]).reshape(tmp[VALUE_INPUT_SIZE + 2].shape) <= 10) | shipyard_cells)

        # self position is zeroed
        mask[:, h, w] = 0
        # spawn is valid except not enough kore
        mask[0] = 1
        if me.kore // board.configuration.spawn_cost < 1:
            mask[0] = 0
        # do_nothing is valid if all masked
        mask[13] = 1-mask[:12].any()
        masks.append(mask)

    return np.stack(res, 0), base, np.stack(masks, 0)


def reverse_obs(obs):
    assert obs.shape[0] == SINGLE_INPUT_SIZE
    obs = copy.deepcopy(obs)
    tmp = copy.deepcopy(obs[2:9])
    obs[2:9] = obs[9:16]
    obs[9:16] = tmp
    return obs


def get_closest_shipyard(board, position, me, excludes=set()):
    min_dist = 1000000
    closest_shipyard = None
    for shipyard in board.shipyards.values():
        if (shipyard.player_id != me.id) or (shipyard.id in excludes):
            continue
        dist = position.distance_to(shipyard.position, board.configuration.size)
        if dist < min_dist:
            min_dist = dist
            closest_shipyard = shipyard
    return closest_shipyard


def get_closest_flight_between_1dim(p0, p1, axis='x'):
    diff = p1 - p0
    d0 = 'E' if axis == 'x' else 'N'
    d1 = 'W' if axis == 'x' else 'S'
    if abs(diff) < BOARD_SIZE / 2:
        step = abs(diff) - 1
        if step == 0: step = ''
        plan = f'{d0}{step}' if diff > 0 else f'{d1}{step}'
    else:
        step = BOARD_SIZE - abs(diff) - 1
        if step == 0: step = ''
        plan = f'{d1}{step}' if diff > 0 else f'{d0}{step}'
    return plan


def generate_launch_action(shipyard, pos1, build=False, horizontal_first=None, boomerang=None, nships=None):
    plan = ''
    pos0 = shipyard.position
    x_diff = pos1.x - pos0.x
    y_diff = pos1.y - pos0.y
    if horizontal_first is None:
        horizontal_first = 0 if np.random.rand() < 0.5 else 1
    if boomerang is None:
        boomerang = 0 if np.random.rand() < 0.5 else 1
    if nships is None:
        nships = np.random.choice([0,1,2])
    assert (x_diff != 0) or (y_diff != 0)
    if x_diff == 0:
        plan += get_closest_flight_between_1dim(pos0.y, pos1.y, axis='y')
        if build:
            plan += 'C'
        plan += get_closest_flight_between_1dim(pos1.y, pos0.y, axis='y')
    elif y_diff == 0:
        plan += get_closest_flight_between_1dim(pos0.x, pos1.x, axis='x')
        if build:
            plan += 'C'
        plan += get_closest_flight_between_1dim(pos1.x, pos0.x, axis='x')
    else:
        if horizontal_first:
            # start with x axis
            plan += get_closest_flight_between_1dim(pos0.x, pos1.x, axis='x')
            plan += get_closest_flight_between_1dim(pos0.y, pos1.y, axis='y')
            if build:
                plan += 'C'
            if boomerang:
                plan += get_closest_flight_between_1dim(pos1.y, pos0.y, axis='y')
                plan += get_closest_flight_between_1dim(pos1.x, pos0.x, axis='x')
            else:
                plan += get_closest_flight_between_1dim(pos1.x, pos0.x, axis='x')
                plan += get_closest_flight_between_1dim(pos1.y, pos0.y, axis='y')
        else:
            # start with y axis
            plan += get_closest_flight_between_1dim(pos0.y, pos1.y, axis='y')
            plan += get_closest_flight_between_1dim(pos0.x, pos1.x, axis='x')
            if build:
                plan += 'C'
            if boomerang:
                plan += get_closest_flight_between_1dim(pos1.x, pos0.x, axis='x')
                plan += get_closest_flight_between_1dim(pos1.y, pos0.y, axis='y')
            else:
                plan += get_closest_flight_between_1dim(pos1.y, pos0.y, axis='y')
                plan += get_closest_flight_between_1dim(pos1.x, pos0.x, axis='x')
    if plan[-1] not in ['N', 'E', 'S', 'W']:
        if plan[-2] == '1':
            plan = plan[:-2]
        else:
            plan = plan[:-1]
    if build:
        min_, max_ = 50, shipyard.ship_count
    else:
        min_, max_ = get_min_ships(len(plan)), shipyard.ship_count
    if max_ >= min_:
        n_launch_ships = {
            0: int(min_),
            1: int(min_*0.5 + max_*0.5),
            2: int(max_)
        }
        next_action = str(ShipyardAction.launch_fleet_with_flight_plan(n_launch_ships[nships], plan))
    else:
        next_action = ''
    return next_action


def get_min_ships(flight_plan_length):
    """
    Number Ships	Max Flight Plan Length
    1	1
    2	2
    3	3
    5	4
    8	5
    13	6
    21	7
    34	8
    55	9
    91	10
    149	11
    245	12
    404	13
    """
    return int(np.ceil(np.exp((flight_plan_length-1) / 2)))


def process_action(action, obs, env_config):
    """
    action: [n_shipyards]
    0. spawn
    1. launch_min_horizontal
    2. launch_min_vertical
    3. launch_middle_horizontal
    4. launch_middle_vertical
    5. launch_max_horizontal
    6. launch_max_vertical
    7. build_min_horizontal
    8. build_min_vertical
    9. build_middle_horizontal
    10. build_middle_vertical
    11. build_max_horizontal
    12. build_max_vertical
    13. do_nothing
    """
    board = Board(obs, env_config)
    me = board.current_player
    n_shipyards = len(me.shipyards)
    assert len(action) == n_shipyards

    spawn_cost = board.configuration.spawn_cost
    kore_left = me.kore

    next_action = {shipyard.id: '' for shipyard in me.shipyards}

    for i, shipyard in enumerate(me.shipyards):

        type, position = np.divmod(action[i], BOARD_SIZE * BOARD_SIZE)
        target_position = Point.from_index(int(position), BOARD_SIZE)
        if type == 13:
            # do_nothing
            continue
        elif (type == 0) or (shipyard.position.distance_to(target_position, BOARD_SIZE) == 0):
            # spawn
            n_spawns = int(min(kore_left // spawn_cost, shipyard.max_spawn))
            if n_spawns > 0:
                next_action[shipyard.id] = str(ShipyardAction.spawn_ships(n_spawns))
                kore_left -= n_spawns * spawn_cost
        else:
            build, tmp = np.divmod(type-1, 3*2)
            nships, vertical_first = np.divmod(tmp, 2)
            next_action[shipyard.id] = generate_launch_action(shipyard, target_position, build=build, horizontal_first=1-vertical_first, boomerang=False, nships=nships)

    return next_action


def get_next_step_fleet_position(board, fleet_uid):
    fleet = copy.deepcopy(board).fleets[fleet_uid]
    direction = fleet.direction
    if fleet.flight_plan:
        if fleet.flight_plan[0] == 'C':
            return fleet.position.to_index(BOARD_SIZE)
        else:
            if fleet.flight_plan[0] in ['N', 'E', 'S', 'W']:
                direction = Direction.from_char(fleet.flight_plan[0])
            elif fleet.flight_plan[0] == '0':
                if fleet.flight_plan[1] == 'C':
                    return fleet.position.to_index(BOARD_SIZE)
                direction = Direction.from_char(fleet.flight_plan[1])
    return (fleet.position.translate(direction.to_point(), BOARD_SIZE)).to_index(BOARD_SIZE)


def deprocess_action(action, obs, conf):
    board = Board(obs, conf)
    orig_board = copy.deepcopy(board)
    action = {k: v for k, v in action.items() if k in board.shipyards.keys()}
    info = {k.id: [None, 0, None, None] for k in board.current_player.shipyards}  # {shipyard_id: [fleet_id, max_distance, target_point_index, last_point_index]}
    uid = 1
    for k, v in action.items():
        board.shipyards[k].next_action = ShipyardAction.from_str(v)
        if v.startswith('LAUNCH'):
            info[k][0] = f'{board.observation["step"] + 1}-{uid}'
            direction = Direction.from_char(v.split('_')[-1][0])
            info[k][3] = (board.shipyards[k].position.translate(direction.to_point(), BOARD_SIZE)).to_index(BOARD_SIZE)
            uid += 1
    player = board.observation['player']
    board = board.next()
    counter = 0
    while any([fleet_uid in board.observation['players'][player][2].keys() for fleet_uid, _, _, _ in info.values()]):
        if counter > 100:
            break
        for shipyard_id, (fleet_uid, max_distance, _, _) in info.items():
            if fleet_uid in board.observation['players'][player][2].keys():
                fleet_position_index = board.observation['players'][player][2][fleet_uid][0]
                distance = Point.from_index(fleet_position_index, BOARD_SIZE).distance_to(orig_board.shipyards[shipyard_id].position, BOARD_SIZE)
                if distance > max_distance:
                    info[shipyard_id][1] = distance
                    info[shipyard_id][2] = fleet_position_index
                info[shipyard_id][3] = get_next_step_fleet_position(board, fleet_uid)
        board = board.next()
        counter += 1
    res = []
    for shipyard_id, (fleet_uid, max_distance, target_point_index, last_point_index) in info.items():
        if shipyard_id in action.keys():
            actstr = action[shipyard_id]
            if actstr.startswith('LAUNCH'):
                flight_plan = actstr.split('_')[2]
                n_launch = int(actstr.split('_')[1])
                if 'C' in flight_plan:
                    build = 1
                    min_ = 50
                else:
                    build = 0
                    min_ = get_min_ships(len(flight_plan))
                if flight_plan[0] in ['E', 'W']:
                    vertical_first = 0
                else:
                    vertical_first = 1
                max_ = orig_board.shipyards[shipyard_id].ship_count
                nships = np.argmin(np.abs(np.array([min_, min_*0.5+max_*0.5, max_]) - n_launch))
                type = 1 + build * 3 * 2 + nships * 2 + vertical_first
            elif actstr.startswith('SPAWN'):
                type = 0
            else:
                type = 13  # do_nothing
        else:
            type = 13  # do_nothing
        to_append = type * (BOARD_SIZE ** 2)
        if fleet_uid is None:
            to_append += orig_board.shipyards[shipyard_id].position.to_index(BOARD_SIZE)
        elif orig_board.shipyards[shipyard_id].position.distance_to(Point.from_index(last_point_index, BOARD_SIZE), BOARD_SIZE) == 0:
            to_append += target_point_index
        else:
            to_append += last_point_index
        res.append(to_append)

    return np.array(res, dtype=np.float32)