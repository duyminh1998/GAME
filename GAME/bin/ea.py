# Author: Minh Hua
# Date: 11/1/2022
# Purpose: This module contains the main evolutionary algorithm that will be used to evolve inter-task mappings.

import random

from GAME.bin.intertask_mappings import *
from GAME.utils.config import config

class GAME:
    """Genetic Algorithms for Mapping Evolution evolves a population of inter-task mapping for transfer learning in reinforcement learning."""
    def __init__(self,
        src_state_var_names:list,
        src_action_names:list,
        src_action_values:list,
        target_state_var_names:list,
        target_action_names:list,
        target_action_values:list,
        keep_top_k:int=1,
        eval_metric:str='avg_fitness',
        pop_size:int=1000,
        crossover_rate:float=0.8,
        mutation_rate:float=0.2,
        sel_strat:str='tournament',
        init_strat:str='random',
        max_evol_gen:int=1000
    ) -> None:
        """
        Description:
            Initializes an instance of the Genetic Algorithms for Mapping Evolution model.

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
            pop_size: the size of the population.
            crossover_rate: the crossover rate.
            mutation_rate: the mutation rate.
            sel_strat: the crossover strategy.
                'tournament': tournament selection.
            init_strat: the initialization strategy.
                'random': uniformly initialize the individual chromosomes.
            max_evol_gen: the maximum number of generations to evolve.

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

        # evolutionary algorithm parameters
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.sel_strat = sel_strat
        self.init_strat = init_strat
        self.max_evol_gen = max_evol_gen

        # load config data
        self.config = config()

    def init_pop(self, strategy:str='random') -> None:
        """
        Description:
            Initializes a population of inter-task mappings.

        Arguments:
            strategy: the strategy used to initialize the population.
                'random': uniformly initialize the individual chromosomes.

        Return:
            (None)
        """
        population = []
        if self.init_strat == 'random': # randomly generate mappings. Each mapping is sampled uniformly
            for _ in range(self.pop_size):
                # first generate the state mapping
                state_mapping_chrom = [random.randint(0, len(self.src_state_var_names) - 1) for _ in range(len(self.target_state_var_names))]
                # then generate the action mapping
                action_mapping_chrom = [random.randint(0, len(self.src_actions) - 1) for _ in range(len(self.target_action_names))]
                intertask_mapping_individual = IntertaskMapping(state_mapping_chrom, action_mapping_chrom, self.src_state_var_names, self.src_action_names, self.target_state_var_names, self.target_action_names)
                # append individual to the population
                population.append(intertask_mapping_individual)
        # save initial population
        self.population = population

    def select_parents(self, num_pairs:int, strategy='tournament', tournament_sel_k:int=None) -> list:
        """
        Description:
            Select parents for crossover according to some strategy.

        Arguments:
            num_pairs: the number of pairs of parents to select. Determines the number of generated offspring.
            strategy: the strategy used to crossover the parents.
                'tournament': tournament selection

        Return:
            (list) a list containing pairs of parents for reproduction.
        """
        parents = []
        if self.sel_strat == 'tournament':
            parents = random.choices(ea.population, k = tournament_sel_k)
            parents = sorted(parents, key=lambda agent: agent.fitness, reverse=True)
        
        return parents

    def crossover(self, strategy='one-pt') -> None:
        """
        Description:
            Generate a number of offspring using certain crossover strategies.

        Arguments:
            strategy: the strategy used to crossover the parents.
                'one-pt': one-point crossover.

        Return:
            (None)
        """

if __name__ == '__main__':
    from GAME.utils.config import *

    # load the config data
    config_data = config()

    # evolution parameters
    src_state_var_names = config_data['MC2D_state_names']
    src_action_names = config_data['MC2D_action_names']
    src_action_values = config_data['MC2D_action_values']
    target_state_var_names = config_data['MC3D_state_names']
    target_action_names = config_data['MC3D_action_names']
    target_action_values = config_data['MC3D_action_values']

    init_pop_size = 100
    crossover_rate = 0.8
    mutation_rate = 0.2
    crossover_strat = 'tournament'
    max_evol_iter = 1000

    # helper variables
    # transforming src data
    src_data_path = config['output_path'] + "10242022 Initial Samples Collection for 2D MC\\test.csv"
    src_data_df = pd.read_csv(src_data_path, index_col = False)
    transformed_df_col_names = config['3DMC_full_transition_df_col_names']

    # evaluation using networks
    network_folder_path = config['pickle_path'] + "11012022 3DMC Neural Nets\\"
    eval_networks = EvaluationNetworks(network_folder_path)
    transformed_df_current_state_cols = config['3DMC_current_state_transition_df_col_names']
    transformed_df_next_state_cols = config['3DMC_next_state_transition_df_col_names']

    ea = GAME(src_state_var_names, src_action_names, target_state_var_names, target_action_names, init_pop_size, crossover_rate, mutation_rate, crossover_strat, max_evol_iter)
    # initialize initial population
    ea.init_pop()
    # evaluate initial population's fitness
    for mapping in ea.population:
        # use mapping to create transformed df
        transformed_df = transform_source_dataset(src_data_df, mapping, transformed_df_col_names, target_action_values)
        # evaluate mapping using the transformed df
        eval_scores = evaluate_mapping(mapping, transformed_df, eval_networks, transformed_df_current_state_cols, transformed_df_next_state_cols, target_action_values)
        # consolidate into one single score
        mapping.fitness = parse_mapping_eval_scores(eval_scores)
        # print debug info
        print('State mapping: {}, Action mapping: {}, Fitness: {}'.format(mapping.state_mapping, mapping.action_mapping, mapping.fitness))

    # main evolution loop
    for gen in ea.max_evol_iter:
        # select parents for crossover
        parents = ea.select_parents()
        # generate offspring using selected parents

        # mutate offspring

        # evaluate offspring fitness

        # replace population with offspring
    
    print('Done!')