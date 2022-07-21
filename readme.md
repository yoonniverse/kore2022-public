Source code for training RL agent for [Kore-2022 game](https://www.kaggle.com/competitions/kore-2022).

## Getting Started

**Tested on Linux**

*Change `num_cpus`/`num_workers` hyperparameter in `configs.py`, `test.py`, `preprocess_data.py`*

1. `conda crate -n kore2022 python=3.8`
2. `conda activate kore2022`
3. `bash install_dependencies.sh`
4. `python download_kaggle_episodes.py`
5. `python preprocess_data.py`
6. `python imitate.py`
7. `python train.py`, while training, check `training_analysis.ipynb` & `result_analysis.ipynb`, interrupt when you wish.
8. `python test.py`
9. Submission to Kaggle: `!tar -czf submission.tar.gz *` and submit tar.gz file to https://www.kaggle.com/competitions/kore-2022/submissions

## Baseline Agents

can be imported from `test.py`

* random_agent: `Agent()` with randomly initialized weights
* balanced_agent: The best agent among those provided by kaggle at https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kore_fleets/kore_fleets.py.
* beta_agent: Winner of beta competition (https://github.com/w9PcJLyb/kore-beta-bot, https://www.kaggle.com/code/solverworld/kore-beta-1st-place-solution)

## Files

1. `configs.py`: define training hyperparameters
2. `train.py`
   * wandb loggging
   * save recent agent as `agent.pth`
   * save agent as `agent{i}.pth` every `save_every` epochs.
   * save recent trajectory informations at `trajectory.jl`, `trajectory_env.jl` for monitoring/debugging
3. `agent.py`: act, learn method of agent / agent has its own trajectory buffer
4. `env.py`: step=>(obs, reward, done), reset=>(obs, reward, done) 함수
5. `net.py`: actorcritic neural network
6. `buffer.py`: ppo buffer
7. `processors.py`: functions to process observation and action from network output to environment interpretable form
8. `test.py`: get winrate of `agent.pth` against baseline agents
9. `utils.py`: global variables and general utility functions
10. `main.py`: used when submitting to Kaggle.
11. `render.py`: https://www.kaggle.com/competitions/kore-2022/discussion/323495
12. `beta_agent.py`: Winner of beta competition (https://github.com/w9PcJLyb/kore-beta-bot, https://www.kaggle.com/code/solverworld/kore-beta-1st-place-solution)

## Observation

`env.step(action)[0]['observation']`, in the format of
```
{
'remainingOverageTime': int,
'step': int, 
'player': 0 or 1, 
'kore': list(441), 
'players': [
    [kore, {shipyardkey: [pos, n_ship, turns_controlled]}, {fleetkey: [pos, kore, n_ship, direction, remaining_plan]}
for each player]
}
```
*pos=(row * size + column)*


## Input Space

### Policy Input

input for policy(actor) network  

`(18+3)x21x21` per shipyard (`n_shipyardx(18+3)x21x21` per step)

1. kore
2. step
3. shipyard
4. shipyard_ship_count
5. shipyard_turns_controlled
6. fleet_kore
7. fleet_ship_count
8. broadcasted_possessing_kore
9. broadcasted_total_asset
10. opponent_shipyard
11. opponent_shipyard_ship_count
12. opponent_shipyard_turns_controlled
13. opponent_fleet_kore
14. opponent_fleet_ship_count
15. opponent_broadcasted_possessing kore 
16. opponent_broadcasted_total_asset
17. shipyard_ships_after_being_attacked 
18. opponent shipyard_ships_after_being_attacked

3 shipyard features

19. shipyard location 
20. 20broadcasted shipyard ship count
21. expected kores mined per step
22. broadcasted shipyard ships after being attacked

### Value Input

input for value(critic) network

`18x21x21`

## Action Space

`14x21x21`

for each position

1. spawn
2. launch_min_horizontal
3. launch_min_vertical
4. launch_middle_horizontal
5. launch_middle_vertical
6. launch_max_horizontal
7. launch_max_vertical
8. build_min_horizontal
9. build_min_vertical
10. build_middle_horizontal
11. build_middle_vertical
12. build_max_horizontal
13. build_max_vertical
14. do_nothing

*mask invalid actions by replacing corresponding logits with -inf*

*mask some actions with heuristic*

=> process_action => `SPAWN_X` or `LAUNCH_X_<FLIGHT_PLAN>`
