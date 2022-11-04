# Author: Minh Hua
# Date: 11/3/2022
# Purpose: This module contains the brute force approach to generating an inter-task mapping.

from GAME.bin.intertask_mappings import *
# from GAME.utils.config import config

from itertools import product

class MASTER:
    """
    Implementation of the MASTER algorithm presented by Matthew E. Taylor, Gregory Kuhlmann, and Peter Stone
    in Taylor, M. E., Kuhlmann, G., & Stone, P. (2008, May). Autonomous transfer for reinforcement learning. In AAMAS (1) (pp. 283-290).
    """
    def __init__(self,
        src_state_var_names:list,
        src_action_names:list,
        src_action_values:list,
        target_state_var_names:list,
        target_action_names:list,
        target_action_values:list,
        src_task_data_folder_and_filename:str,
        neural_networks_folder:str,
        keep_top_k:int=1,
        eval_metric:str='avg_fitness'
    ) -> None:
        """
        Description:
            Initializes an instance of the MASTER model.

        Arguments:
            src_state_var_names: the name of the state variables in the source task.
            src_action_names: the name of the actions in the source task.
            src_action_values: the numerical value of the actions in the source task.
            target_state_var_names: the name of the state variables in the target task.
            target_action_names: the name of the actions in the target task.
            target_action_values: the numerical value of the actions in the target task.
            keep_top_k: return the top k intertask mapping individuals.
            eval_metric: the metric to determine the fitness of individuals.
                'avg_fitness': average the fitness values across the predicted states and actions.

        Return:
            (None)
        """
        # save class variables
        self.src_state_var_names = src_state_var_names
        self.src_action_names = src_action_names
        self.src_action_values = src_action_values
        self.target_state_var_names = target_state_var_names
        self.target_action_names = target_action_names
        self.target_action_values = target_action_values
        self.keep_top_k = keep_top_k
        self.eval_metric = eval_metric

        # assign a unique code to each state variable and action for efficiency
        self.src_state_codes = {src_state_variable : id for src_state_variable, id in zip(src_state_var_names, range(len(src_state_var_names)))}
        self.src_action_codes = {src_action : id for src_action, id in zip(src_action_names, src_action_values)}
        self.target_state_codes = {target_state_variable : id for target_state_variable, id in zip(target_state_var_names, range(len(target_state_var_names)))}
        self.target_action_codes = {target_action : id for target_action, id in zip(target_action_names, target_action_values)}

        # load config data
        # self.config = config()

        # transforming src data for mapping evaluation
        src_data_path = config['output_path'] + src_task_data_folder_and_filename
        self.src_data_df = pd.read_csv(src_data_path, index_col = False)
        self.transformed_df_col_names = config['3DMC_full_transition_df_col_names']
        self.transformed_df_current_state_cols = config['3DMC_current_state_transition_df_col_names']
        self.transformed_df_next_state_cols = config['3DMC_next_state_transition_df_col_names']

        # evaluation using networks
        network_folder_path = config['pickle_path'] + neural_networks_folder
        self.eval_networks = EvaluationNetworks(network_folder_path)

    def evolve(self) -> list:
        """
        Description:
            Run the MASTER model and generate a list of intertask mappings.

        Arguments:
            

        Return:
            (list) a list of the most fit intertask mappings.
        """
        population = []
        # enumerate all possible intertask mappings
        for state_mapping in product(self.src_state_codes.values(), repeat = len(self.target_state_var_names)): # all possible state mappings
            for action_mapping in product(self.src_action_codes.values(), repeat = len(self.target_state_var_names)): # all possible action mappings
                mapping = IntertaskMapping(state_mapping, action_mapping, self.src_state_var_names, self.src_action_names, self.target_state_var_names, self.target_action_names)
                # evaluate the mapping
                # use mapping to create transformed df
                transformed_df = transform_source_dataset(self.src_data_df, mapping, self.transformed_df_col_names, self.target_action_values)
                # evaluate mapping using the transformed df
                eval_scores = evaluate_mapping(mapping, transformed_df, self.eval_networks, self.transformed_df_current_state_cols, self.transformed_df_next_state_cols, self.target_action_values)
                # consolidate into one single score
                mapping.fitness = parse_mapping_eval_scores(eval_scores)
                # print debug info
                print('State mapping: {}, Action mapping: {}, Fitness: {}'.format(mapping.state_mapping, mapping.action_mapping, mapping.fitness))
                # save mapping to population
                population.append(mapping)

        # return the k individuals with the highest fitness
        return sorted(population, key=lambda agent: agent.fitness, reverse=True)[:self.keep_top_k]

if __name__ == "__main__":

    MC2D_states = ['x_position', 'x_velocity']
    MC3D_states = ['x_position', 'x_velocity', 'y_position', 'y_velocity']
    MC2D_actions = ['Left', 'Neutral', 'Right']
    MC3D_actions = ['Neutral', 'West', 'East', 'South', 'North']

    src_state_var_names = MC2D_states
    src_action_names = MC2D_actions
    src_action_values = [0, 1, 2]
    target_state_var_names = MC3D_states
    target_action_names = MC3D_actions
    target_action_values = [0, 1, 2, 3, 4]
    src_task_data_folder_and_filename = "10242022 Initial Samples Collection for 2D MC\\test.csv"
    neural_networks_folder = "11012022 3DMC Neural Nets\\"

    keep_top_k = 1
    eval_metric = 'avg_fitness'

    master = MASTER(src_state_var_names, src_action_names, src_action_values, target_state_var_names, target_action_names, target_action_values, 
    src_task_data_folder_and_filename, neural_networks_folder, keep_top_k, eval_metric)
    final_mappings = master.evolve()