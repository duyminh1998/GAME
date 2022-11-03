# Author: Minh Hua
# Date: 11/1/2022
# Purpose: This module contains code to support inter-task mappings.

import pandas as pd
import numpy as np
import os
from sklearn.neural_network import MLPRegressor
import pickle

class Mapping:
    """A class that represents a generic mapping. Can be either a state or action mapping."""
    def __init__(self, type:str, keys:list, values:list) -> None:
        """
        Description:
            Initializes a Mapping.

        Arguments:
            type: the type of mapping. Must be either 'state' or 'action'.
            keys: the keys of the map.
            values: the corresponding values of the map.

        Return:
            (None)
        """
        self.type = type
        assert len(keys) == len(values)
        self.mapping = {k : v for k, v in zip(keys, values)}

class IntertaskMapping:
    """A class that represents an inter-task mapping consisting of a state mapping and action mapping."""
    def __init__(self, 
        state_mapping:list,
        action_mapping:list,
        src_state_var:list,
        src_actions:list,
        target_state_var:list,
        target_actions:list,
        fitness:float=None
    ):
        """
        Description:
            Initializes an inter-task mapping.

        Arguments:
            state_mapping: a list of values indicating the state mapping. Treat this as the state mapping chromosome.
            action_mapping: a list of values indicating the action mapping. Treat this as the action mapping chromsome.
            src_state_var: a list of the source task state variables in plain text. Used to decode the chromosome.
            src_actions: a list of the source task actions in plain text. Used to decode the chromosome.
            target_state_var: a list of the target task state variables in plain text. Used to decode the chromosome.
            target_actions: a list of the target actions in plain text. Used to decode the chromosome.
            fitness: the initial fitness of the mapping.

        Return:
            (None)
        """
        # intertask mapping attributes
        self.state_mapping = state_mapping
        self.action_mapping = action_mapping
        self.fitness = None

        # assign meaning to the state and action mappings
        # note that the map goes from the target to the source
        self.decoded_state_mapping = {target_state_var[i] : src_state_var[j] for i, j in zip(range(len(target_state_var)), state_mapping)}
        self.decoded_action_mapping = {target_actions[i] : src_actions[j] for i, j in zip(range(len(target_actions)), action_mapping)}
        self.multiple_mapped_actions = [] # get a list of multiple target actions that a single src task action gets mapped to
        for src_action in range(len(src_actions)):
            self.multiple_mapped_actions.append([idx for idx in range(len(action_mapping)) if action_mapping[idx] == src_action])

class EvaluationNetworks:
    """This class contains all the neural networks that are used to evaluate an intertask mapping's transformed data."""
    def __init__(self, nn_folder_path:str) -> None:
        """
        Description:
            Initializes a class that holds all the neural networks for evaluating intertask mappings.

        Arguments:
            nn_folder_path: the path containing the pickled neural networks. Assumes that these networks ahve already been trained.

        Return:
            (None)
        """
        self.networks = {}
        # read in the neural networks
        for file in os.listdir(nn_folder_path):
            if file.endswith(".pickle"):
                # create a key for the network which is the action and the predicted state
                split_file = file.split('-')
                nn_key = split_file[0][1] + '_' + split_file[1][1:].split('.')[0]
                with open(os.path.join(nn_folder_path, file), 'rb') as f:
                    mlp = pickle.load(f)
                    self.networks[nn_key] = mlp

    def get_network(self, action:int, state:str) -> MLPRegressor:
        """
        Description:
            Returns the neural network that predicts the state given the action.

        Arguments:
            action: the action that the neural network approximates.
            state: the state that the neural network predicts.

        Return:
            (MLPRegressor) the neural network that predicts the state.
        """
        return self.networks[str(action) + '_' + state]

def transform_source_dataset(src_dataset:pd.DataFrame, intertask_map:IntertaskMapping, target_col_names:list, target_actions:list) -> pd.DataFrame:
    """
    Description:
        Transforms a dataset of transitions from the source task into a dataset of transitions in the target task.

    Arguments:
        src_dataset: the dataset containing transition samples in the source task.
        intertask_map: the intertask mapping to transform the samples.
        target_col_names: the names of the columns in the target task (transformed) dataset.
        target_actions: a list of actions in the target task.

    Return:
        (pd.DataFrame) a dataset of transformed transition samples from the source task.
    """
    # # temp list to hold data before conversion to dataframe
    # transformed_src_data = []

    # # save column information for reuse
    # parsed_col_names_arr = [col.split('_') for col in transformed_df_col_names]
    # current_or_next_arr = [col[0] for col in parsed_col_names_arr]
    # state_or_action_arr = [col[1] for col in parsed_col_names_arr]

    # # loop through the source dataset and transform each row
    # for _, row in src_dataset.iterrows():
    #     # build mapped state data first
    #     src_current_state = []
    #     src_next_state = []
    #     # loop through the columns and transform them
    #     for col_idx in range(len(transformed_df_col_names)):
    #         # ignore actions for now
    #         if not state_or_action_arr[col_idx] == 'action':
    #             # we are looking at s or s'
    #             # construct the mapped state
    #             reconstructed_col_name = "_".join(parsed_col_names_arr[col_idx][1:])
    #             # transform column name into source column using mapping
    #             src_task_col_name = intertask_map.decoded_state_mapping[reconstructed_col_name]
    #             if current_or_next_arr[col_idx] == 'Current':
    #                 src_current_state.append(row['Current_' + src_task_col_name])
    #             elif current_or_next_arr[col_idx] == 'Next':
    #                 src_next_state.append(row['Next_' + src_task_col_name])
    #     # for each mapped action, we must create a new (s, a, s') tuple
    #     for mapped_action in intertask_map.multiple_mapped_actions[int(row['Current_action'])]:
    #         transformed_data_point = src_current_state + [mapped_action] + src_next_state
    #         transformed_src_data.append(transformed_data_point)
    # # output the data to a DataFrame
    # return pd.DataFrame(transformed_src_data, columns = transformed_df_col_names)

    # init empty dataframe for transformed src data
    transformed_src_df = pd.DataFrame()

    # transform state data first
    for col_name in target_col_names:
        # copy all state columns and ignore actions for now
        split_col_names = col_name.split('_')
        current_or_next = split_col_names[0]
        state_or_action = split_col_names[1]
        # check that we are not looking at the action column
        if not state_or_action == 'action':
            # we are looking at s or s'
            # construct the mapped state
            reconstructed_col_name = "_".join(split_col_names[1:])
            # transform column name into source column using mapping
            src_task_col_name = intertask_map.decoded_state_mapping[reconstructed_col_name]
            # are we looking at s or s'?
            if current_or_next == 'Current' or current_or_next == 'Next':
                transformed_src_df[col_name] = src_dataset[current_or_next + '_' + src_task_col_name]
    
    # create dummy variables to mark multiple mapped actions
    action_dummy_data = np.zeros(shape=(len(target_actions), len(src_dataset)))
    for row_idx, row in src_dataset.iterrows():
        multiple_mapped_actions = intertask_map.multiple_mapped_actions[int(row['Current_action'])]
        for mapped_action in multiple_mapped_actions:
            action_dummy_data[mapped_action][row_idx] = 1
    for action_dummy_idx in range(len(target_actions)):
        transformed_src_df['Current_action_' + str(action_dummy_idx)] = action_dummy_data[action_dummy_idx]
    
    return transformed_src_df.reset_index(drop=True)

def evaluate_mapping(
    mapping:IntertaskMapping,
    transformed_df:pd.DataFrame,
    eval_networks:EvaluationNetworks,
    current_state_cols:list,
    next_state_cols:list,
    actions:list
) -> dict:
    """
    Description:
        Evaluates an intertask mapping using the transformed dataset and the evaluation networks.

    Arguments:
        mapping: the intertask mapping.
        transformed_df: the dataset of transformed source task transitions.
        eval_networks: the set of neural networks to be used for evaluation.
        current_state_cols: the names of the columns that represent the current state.
        next_state_cols: the names of the columns that represent the afterstate.
        actions: a list of actions in the target task.

    Return:
        (pd.DataFrame) a dataset of transformed transition samples from the source task.
    """
    # init a dictionary to hold the evaluation metrics
    eval_scores = {}
    for action in actions:
        # filter the dataset by the action
        action_col_name = 'Current_action_' + str(action)
        src_df_by_action = transformed_df[transformed_df[action_col_name] == 1]

        # evaluate the mapping's transformed df
        features = src_df_by_action[current_state_cols]
        # we evaluate each afterstate variable independently
        for target_name in next_state_cols:
            target = src_df_by_action[target_name]
            eval_mlp = eval_networks.get_network(action, target_name)
            eval_score = eval_mlp.score(features, target)
            eval_scores['{}_{}'.format(action, target_name)] = eval_score
    # return the dictionary of raw scores
    return eval_scores

def parse_mapping_eval_scores(eval_scores:dict, strategy:str='average') -> list:
    """
    Description:
        Convert the intertask mapping evaluation scores into a single score based on different strategies.

    Arguments:
        eval_scores: the dictionary of evaluation scores returned by evaluate_mapping().
        strategy: different ways to convert eval_scores into a single score.
            'average': average all the scores into one score.
            'by_state': return a list of scores aggregated by the next predicted state.
            'by_action': return a list of scores aggregated by the action.

    Return:
        (list) consolidated scores.
    """
    consolidated_scores = []
    if strategy == 'average': # just average all the scores
        consolidated_scores.append(sum(eval_scores.values()) / len(eval_scores.values()))
    
    return consolidated_scores

if __name__ == "__main__":
    MC2D_states = ['x_position', 'x_velocity']
    MC3D_states = ['x_position', 'y_position', 'x_velocity', 'y_velocity']
    MC2D_actions = ['Left', 'Neutral', 'Right']
    MC3D_actions = ['Neutral', 'West', 'East', 'South', 'North']

    # evolution parameters
    src_state_var = MC2D_states
    src_actions = MC2D_actions
    target_state_var = MC3D_states
    target_actions = MC3D_actions

    mapping = IntertaskMapping([1, 1, 0, 0], [2, 0, 2, 2, 2], src_state_var, src_actions, target_state_var, target_actions)

    src_data_path = "C:\\Users\\minhh\\Documents\\JHU\\Fall 2022\\Evolutionary and Swarm Intelligence\\src\\GAME\\output\\10242022 Initial Samples Collection for 2D MC\\test.csv"
    src_data_df = pd.read_csv(src_data_path, index_col = False)
    transformed_df_col_names = ['Current_x_position', 'Current_x_velocity', 'Current_y_position', "Current_y_velocity",
    'Current_action', 'Next_x_position', 'Next_x_velocity', 'Next_y_position', 'Next_y_velocity']

    transformed_df = transform_source_dataset(src_data_df, mapping, transformed_df_col_names, target_actions)
    # transformed_df_out_path = "C:\\Users\\minhh\\Documents\\JHU\\Fall 2022\\Evolutionary and Swarm Intelligence\\src\\GAME\\output\\11022022 Transformed Source Data 3DMC\\transformed_df.csv"
    # transformed_df.to_csv(transformed_df_out_path, index = False)

    network_folder_path = "C:\\Users\\minhh\\Documents\\JHU\\Fall 2022\\Evolutionary and Swarm Intelligence\\src\\GAME\\pickle\\11012022 3DMC Neural Nets\\"
    eval_networks = EvaluationNetworks(network_folder_path)
    # test_mlp = eval_networks.get_network(0, 'Next_x_position')

    actions = [0, 1, 2, 3, 4]
    current_state_cols = ['Current_x_position', 'Current_x_velocity', 'Current_y_position', "Current_y_velocity"]
    next_state_cols = ['Next_x_position', 'Next_x_velocity', 'Next_y_position', 'Next_y_velocity']

    eval_results = evaluate_mapping(mapping, transformed_df, eval_networks, current_state_cols, next_state_cols, actions)
    print(parse_mapping_eval_scores(eval_results))