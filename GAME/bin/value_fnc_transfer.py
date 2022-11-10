from GAME.utils.config import config
from GAME.agents.sarsa_lambda import SarsaLambdaCMAC3DMountainCar
from GAME.agents.TileCoding import *
import pickle
import numpy as np

config_data = config()

mc2d_agent_folder_path = config_data['pickle_path'] + '11032022 2DMC Sample Collection 100 Episodes with Training\\'
mc2d_agent_file_path = mc2d_agent_folder_path + '2DMC_100_ep_a1.2_l0.95_e0_nt8.pickle'

mc3d_agent_folder_path = config_data['pickle_path'] + '11072022 Train MC3D No Transfer\\'
mc3d_agent_file_path = mc3d_agent_folder_path + 'trial0_agent_alpha_0.75_lamb_0.99_gam_1.00_eps_0.01_method_replacing_ntiles_8_max_size_4096.pickle'

with open(mc2d_agent_file_path, 'rb') as f:
    mc2d_agent = pickle.load(f)

with open(mc3d_agent_file_path, 'rb') as f:
    mc3d_agent = pickle.load(f)

# agent
alpha = 0.75
lamb = 0.99
gamma = 1
method = 'replacing'
epsilon = 0
num_of_tilings = 1
max_size = 4096
empty_agent = SarsaLambdaCMAC3DMountainCar(alpha, lamb, gamma, method, epsilon, num_of_tilings, max_size)

state_mapping = [0, 1, 0, 1]
action_mapping = [1, 0, 2, 0, 2]

for src_w_idx, src_weight in enumerate(mc2d_agent.weights):
    # find source state and action values corresponding to weight i in hash table
    for k, v in mc2d_agent.hash_table.dictionary.items():
        if v == src_w_idx:
            src_coord = k
            break
    # extract the source state and action values from the coordinates
    tiling_no = src_coord[0]
    src_state_vals = src_coord[1:3]
    src_action_vals = src_coord[-1]
    # get same target state and actions
    target_state_val = [src_state_vals[i] for i in state_mapping]
    target_action_vals = [i for i in range(len(action_mapping)) if action_mapping[i] == src_action_vals]
    for target_action_val in target_action_vals:
        coordinates = (tiling_no, *target_state_val, target_action_val)
        target_w_idx = hashcoords(coordinates, empty_agent.hash_table)
        # set target weight
        empty_agent.weights[target_w_idx] = src_weight

# set all empty weights in target CMAC to be the average value of all non-zero weights in the target CMAC
nonzero_weights_idx = empty_agent.weights.nonzero()
average_weight = np.mean(empty_agent.weights[nonzero_weights_idx])
for w_idx in range(len(empty_agent.weights)):
    if empty_agent.weights[w_idx] == 0:
        empty_agent.weights[w_idx] = average_weight

save_agent_path = config_data['pickle_path'] + '11092022 3DMC agent with initialized weights//agent.pickle'
with open(save_agent_path, 'wb') as f:
    pickle.dump(empty_agent, f)

# print('{}'.format(agent.get_active_tiles([0, -5], 0)))
print('hello')