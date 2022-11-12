# Author: Minh Hua
# Date: 11/11/2022
# Purpose: Runs 15 trials of evolving mappings for Mountain Car.

import os

from GAME.bin.ea import GAME
from GAME.bin.intertask_mappings import *
from GAME.utils.config import config
from GAME.agents.stats_saver import StatisticsSaver, MappingSearchExperimentInfo

# load the config data
config_data = config()

# variables to identify the task
target_task_name = '3DMC'
src_state_var_names = config_data['MC2D_state_names']
src_action_names = config_data['MC2D_action_names']
src_action_values = config_data['MC2D_action_values']
target_state_var_names = config_data['MC3D_state_names']
target_action_names = config_data['MC3D_action_names']
target_action_values = config_data['MC3D_action_values']

# evolution parameters
eval_metric = 'average'
pop_size = 10
crossover_rate = 0.8
mutation_rate = 0.2
init_strat = 'random'
sel_strat = 'tournament'
tournament_sel_k = int(0.4 * pop_size)
crossover_strat = 'fusion'
mutation_strat = 'uniform'
replace_strat = 'replace-all-parents'
top_k_elitism = None
max_evol_gen = 10
early_stop = True
early_stop_gen = 3
early_stop_thresh = 10**-4
print_debug = True

save_every = 1
agent_params = {
    'eval_metric': eval_metric,
    'pop_size': pop_size,
    'crossover_rate': crossover_rate,
    'mutation_rate': mutation_rate,
    'init_strat': init_strat,
    'sel_strat': sel_strat,
    'tournament_sel_k': tournament_sel_k,
    'crossover_strat': crossover_strat,
    'mutation_strat': mutation_strat,
    'replace_strat': replace_strat,
    'top_k_elitism': top_k_elitism,
    'max_evol_gen': max_evol_gen,
    'early_stop': early_stop,
    'early_stop_gen': early_stop_gen,
    'early_stop_thresh': early_stop_thresh
}
search_exp_info = MappingSearchExperimentInfo('2DMC', '3DMC', 'GAME', agent_params)

standard_features = False
standard_targets = False

count_comparisons = True

# helper variables
# transforming src data
src_task_data_folder_and_filename = os.path.join("11032022 2DMC Sample Collection 100 Episodes with Training", "2DMC_100_episodes_sample_data.csv")
neural_networks_folder = "11012022 3DMC Neural Nets"

# run for 15 trials
trials = 15
for trial in range(trials):
    print("Trial: {}".format(trial))
    
    # trial-specific params
    save_output_path  = os.path.join(config_data['output_path'], '11112022 MC EA', 'MC_trial{}_results.txt'.format(trial))
    stats_saver = StatisticsSaver(search_exp_info, trial, True)
    stats_folder_path = os.path.join(config_data['output_path'], '11112022 MC EA')
    stats_filename = 'MC_trial{}_stats.txt'.format(trial)
    stats_pickle = 'MC_trial{}_stats.pickle'.format(trial)

    # run the GAME model
    ea = GAME(target_task_name, src_state_var_names, src_action_names, src_action_values, target_state_var_names, target_action_names, target_action_values, 
    src_task_data_folder_and_filename, neural_networks_folder, eval_metric, pop_size, crossover_rate, 
    mutation_rate, init_strat, sel_strat, tournament_sel_k, crossover_strat, mutation_strat, replace_strat, top_k_elitism, max_evol_gen, 
    early_stop, early_stop_gen, early_stop_thresh, print_debug, save_output_path, save_every, stats_saver, stats_folder_path, stats_filename, standard_features, standard_targets, count_comparisons)
    ea.evolve()
    
    # print the final evolved population
    for mapping in ea.population:
        print(mapping)

    with open(os.path.join(stats_folder_path, stats_pickle), 'wb') as f:
        pickle.dump(ea.stats_saver, f)