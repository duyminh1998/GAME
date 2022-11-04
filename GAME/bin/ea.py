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
        src_task_data_folder_and_filename:str,
        neural_networks_folder:str,
        keep_top_k:int=1,
        eval_metric:str='average',
        pop_size:int=1000,
        crossover_rate:float=0.8,
        mutation_rate:float=0.2,
        init_strat:str='random',
        sel_strat:str='tournament',
        tournament_select_k:int=5,
        crossover_strat:str='one-pt',
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
            src_task_data_folder_and_filename: the folder and filename of the transition samples in the source task.
            neural_networks_folder: the folder containing the neural networks being used for evaluation.
            keep_top_k: return the top k intertask mapping individuals.
            eval_metric: the metric to determine the fitness of individuals.
                'average': average the fitness values across the predicted states and actions.
            pop_size: the size of the population.
            crossover_rate: the crossover rate.
            mutation_rate: the mutation rate.
            init_strat: the initialization strategy.
                'random': uniformly initialize the individual chromosomes.
            sel_strat: the crossover strategy.
                'tournament': tournament selection.
            tournament_sel_k: the number of individuals to include in a tournament.
            crossover_strat: the crossover strategy.
                'one-pt': one-point crossover.
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
        self.init_strat = init_strat
        self.sel_strat = sel_strat
        self.tournament_sel_k = tournament_sel_k
        self.crossover_strat = crossover_strat
        self.max_evol_gen = max_evol_gen

        # load config data
        self.config_data = config()

        # transforming src data for mapping evaluation
        src_data_path = self.config_data['output_path'] + src_task_data_folder_and_filename
        self.src_data_df = pd.read_csv(src_data_path, index_col = False)
        self.transformed_df_col_names = self.config_data['3DMC_full_transition_df_col_names']
        self.transformed_df_current_state_cols = self.config_data['3DMC_current_state_transition_df_col_names']
        self.transformed_df_next_state_cols = self.config_data['3DMC_next_state_transition_df_col_names']

        # evaluation using networks
        network_folder_path = self.config_data['pickle_path'] + neural_networks_folder
        self.eval_networks = EvaluationNetworks(network_folder_path)

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
                action_mapping_chrom = np.random.choice(self.src_action_values, size = len(self.target_action_names), replace = True)
                intertask_mapping_individual = IntertaskMapping(state_mapping_chrom, action_mapping_chrom, self.src_state_var_names, self.src_action_names, self.target_state_var_names, self.target_action_names)
                # append individual to the population
                population.append(intertask_mapping_individual)
        # save initial population
        self.population = population

    def select_parents(self, strategy='tournament', tournament_sel_k:int=None) -> IntertaskMapping:
        """
        Description:
            Select parents for crossover according to some strategy.

        Arguments:
            strategy: the strategy used to crossover the parents.
                'tournament': tournament selection

        Return:
            (IntertaskMapping) a single IntertaskMapping parent chosen from a selection method.
        """
        if self.sel_strat == 'tournament':
            parents = random.choices(ea.population, k = tournament_sel_k)
            parent = sorted(parents, key = lambda agent: agent.fitness, reverse=True)
        
        return parent[0] # return one parent

    def crossover(self, parent_1:IntertaskMapping, parent_2:IntertaskMapping, strategy='one-pt') -> IntertaskMapping:
        """
        Description:
            Generate a number of offspring using certain crossover strategies.

        Arguments:
            parent_1: the first parent.
            parent_2: the second parent.
            strategy: the strategy used to crossover the parents.
                'one-pt': one-point crossover.
                'fusion': select the bit from the higher-fitness parent.

        Return:
            (IntertaskMapping) an IntertaskMapping offspring.
        """
        if self.crossover_strat == 'one-pt':
            # randomly select a point to crossover the parents
            pass
        elif self.crossover_strat == 'fusion':
            # set the child's bit to be the bit belonging to the parent with the higher fitness
            # copy the state mapping first
            state_mapping = []
            for parent_1_bit, parent_2_bit in zip(parent_1.state_mapping, parent_2.state_mapping):
                if parent_1_bit == parent_2_bit:
                    state_mapping.append(parent_1_bit)
                elif parent_1.fitness >= parent_2_bit:
                    state_mapping.append(parent_1_bit)
                else:
                    state_mapping.append(parent_2_bit)
            # copy the action mapping last
            action_mapping = []
            for parent_1_bit, parent_2_bit in zip(parent_1.action_mapping, parent_2.action_mapping):
                if parent_1_bit == parent_2_bit:
                    action_mapping.append(parent_1_bit)
                elif parent_1.fitness >= parent_2_bit:
                    action_mapping.append(parent_1_bit)
                else:
                    action_mapping.append(parent_2_bit)
            return IntertaskMapping(state_mapping, action_mapping, self.src_state_var_names, self.src_action_names, self.target_state_var_names, self.target_action_names)

    def evaluate_fitness(self, mapping:IntertaskMapping, set_fitness:float=True) -> float:
        """
        Description:
            Generate a number of offspring using certain crossover strategies.

        Arguments:
            mapping: the IntertaskMapping to evaluate the fitness for.
            set_fitness: wether or not we want to set the individual's fitness after evaluation.

        Return:
            (float) the fitness of the IntertaskMapping individual.
        """
        # use mapping to create transformed df
        transformed_df = transform_source_dataset(self.src_data_df, mapping, self.transformed_df_col_names, self.target_action_values)
        # evaluate mapping using the transformed df
        eval_scores = evaluate_mapping(mapping, transformed_df, self.eval_networks, self.transformed_df_current_state_cols, self.transformed_df_next_state_cols, self.target_action_values)
        consolidated_score = parse_mapping_eval_scores(eval_scores, strategy = self.eval_metric)
        if self.eval_metric == 'average':
            consolidated_score = consolidated_score[0]
        # consolidate into one single score
        if set_fitness:
            mapping.fitness = consolidated_score
        return consolidated_score

    def evolve(self) -> list:
        """
        Description:
            Run the GAME model and generate a list of intertask mappings.

        Arguments:
            

        Return:
            (list) a list of the most fit intertask mappings.
        """
        # initialize initial population
        self.init_pop()
        # evaluate initial population's fitness
        for mapping in self.population:
            mapping.fitness = self.evaluate_fitness(mapping)
            # print debug info
            print('State mapping: {}, Action mapping: {}, Fitness: {}'.format(mapping.state_mapping, mapping.action_mapping, mapping.fitness))

        # main evolution loop
        for gen in range(self.max_evol_gen):
            offspring = []
            # generate a number of offspring
            for _ in range(self.pop_size):
                # select parents for crossover
                parent_1 = self.select_parents(tournament_sel_k = self.tournament_sel_k)
                parent_2 = self.select_parents(tournament_sel_k = self.tournament_sel_k)
                # generate offspring using crossover
                offspring = self.crossover(parent_1, parent_2, self.crossover_strat)

                # mutate offspring

                # evaluate offspring fitness
                offspring.fitness = self.evaluate_fitness(offspring)
                # print debug info
                print('Offspring, State mapping: {}, Action mapping: {}, Fitness: {}'.format(offspring.state_mapping, offspring.action_mapping, offspring.fitness))

            # replace population with offspring


if __name__ == '__main__':
    from GAME.utils.config import *

    # load the config data
    config_data = config()

    # variables to identify the task
    src_state_var_names = config_data['MC2D_state_names']
    src_action_names = config_data['MC2D_action_names']
    src_action_values = config_data['MC2D_action_values']
    target_state_var_names = config_data['MC3D_state_names']
    target_action_names = config_data['MC3D_action_names']
    target_action_values = config_data['MC3D_action_values']

    # evolution parameters
    keep_top_k = 1
    eval_metric = 'average'
    pop_size = 5
    crossover_rate = 0.8
    mutation_rate = 0.2
    init_strat = 'random'
    sel_strat = 'tournament'
    tournament_sel_k = 5
    crossover_strat = 'fusion'
    max_evol_gen = 1000

    # helper variables
    # transforming src data
    src_task_data_folder_and_filename = "11032022 2DMC Sample Collection 100 Episodes with Training\\2DMC_100_episodes_sample_data.csv"
    neural_networks_folder = "11012022 3DMC Neural Nets\\"

    ea = GAME(src_state_var_names, src_action_names, src_action_values, target_state_var_names, target_action_names, target_action_values, 
    src_task_data_folder_and_filename, neural_networks_folder, keep_top_k, eval_metric, pop_size, crossover_rate, mutation_rate, init_strat, sel_strat, tournament_sel_k, crossover_strat, max_evol_gen)
    # run the GAME model
    ea.evolve()