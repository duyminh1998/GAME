# Author: Minh Hua
# Date: 11/7/2022
# Purpose: This script trains neural networks to approximate the transitions for 4v3 Keepaway.

from GAME.utils.data_loaders import TransitionDataLoader
from GAME.utils.config import config
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
import pickle
import os
import json

config_data = config()
file_path = config_data["output_path"] + "11072022 4v3 RCS Sample Data Collection\\keepaway_4v3_transitions_v2.csv"
current_state_cols = config_data['4v3_current_state_transition_df_col_names']
next_state_cols = config_data['4v3_next_state_transition_df_col_names']
action_col_name = config_data['action_transition_df_col_name']
nn_folder_path = config_data["pickle_path"] + "11092022 4v3 Neural Nets\\"

## nn training parameters
parameters = {
    'hidden_layer_sizes': [(40,), (50,), (100,)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.01, 0.1, 0.2],
    'max_iter': [10000]
}

actions = [config_data['4v3_action_values'][1]]
targets = next_state_cols[-3:]
for action in actions:
    data = TransitionDataLoader(file_path, current_state_cols, next_state_cols, action, action_col_name)
    for target in targets:
        print("Evaluating action: {}, target: {}".format(action, target))
        df_with_one_target = data.split_features_targets(target)
        X = df_with_one_target[data.current_state_cols]
        y = df_with_one_target[target]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        mlp = MLPRegressor()
        clf = GridSearchCV(mlp, parameters)

        clf.fit(X_val, y_val)

        # param_file = nn_folder_path + 'a{}--s{}--cv--results.txt'.format(action, 'Next-' + '_'.join(target.split('-')[1:]))
        # with open(param_file, 'r') as f:
        #     network_params = json.loads(f.readline())

        network_params = clf.best_params_

        best_mlp = MLPRegressor(hidden_layer_sizes=network_params['hidden_layer_sizes'], 
            activation=network_params['activation'], 
            learning_rate=network_params['learning_rate'], 
            learning_rate_init=network_params['learning_rate_init'], 
            solver=network_params['solver'], 
            random_state=42, 
        max_iter=network_params['max_iter'] * 3)

        final_mlp = best_mlp.fit(X_train, y_train)

        # save crossval results and model
        nn_cv_results_filename = 'a{}--s{}.txt'.format(action, target)
        nn_model_filename = 'a{}--s{}.pickle'.format(action, target)
        with open(os.path.join(nn_folder_path, nn_cv_results_filename), 'w') as f:
            f.write(json.dumps(network_params))
            f.write('\nScore: {}'.format(final_mlp.score(X_train, y_train)))
        with open(os.path.join(nn_folder_path, nn_model_filename), 'wb') as f:
            pickle.dump(final_mlp, f)

        print(network_params)