from GAME.utils.config import config
from GAME.utils.data_miners import RCSLogMiner
import os
import shutil

config_data = config()

logs_folder = 'logs'

for file in os.listdir(logs_folder):
    if file.endswith('.kwy'):
        experiment_name = file.split('.')[0]
        break
# experiment_name = "202211071930-UbuntuXenialSmall"
rcg_csv_filepath = os.path.join(logs_folder, "{}.rcg.csv".format(experiment_name))
logs_folderpath = os.path.join(logs_folder, experiment_name)
num_keepers = 4
num_takers = 3
num_state_vars = 19
num_actions = 4
transition_df_col_names = ['Cycle', 'ID_kp_w_ball']
transition_df_col_names = transition_df_col_names + config_data['4v3_full_transition_df_col_names']
transition_df_col_dtypes = [
    'int', 'int', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
        'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'int',
        'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float',
        'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float'
]
log_miner = RCSLogMiner(rcg_csv_filepath, logs_folderpath, transition_df_col_names, transition_df_col_dtypes, num_keepers, num_takers, num_state_vars, num_actions)
csv_out_path = logs_folder
csv_out_name = 'keepaway_4v3_transitions_{}.csv'.format(experiment_name)
log_miner.export_data(csv_out_path, csv_out_name)

for file in os.listdir(logs_folder):
    if file.split(".")[0] == experiment_name:
        if file.endswith('rcg') or file.endswith('kwy') or file.endswith('rcg.csv'):
            os.remove(os.path.join(logs_folder, file))