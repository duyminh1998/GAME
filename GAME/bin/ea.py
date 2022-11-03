# Author: Minh Hua
# Date: 11/1/2022
# Purpose: This module contains the main evolutionary algorithm that will be used to evolve inter-task mappings.

import random

from GAME.bin.intertask_mappings import IntertaskMapping

class GAME:
    """Genetic Algorithms for Mapping Evolution evolves a population of inter-task mapping for transfer learning in reinforcement learning."""
    def __init__(self,
        src_state_var:list,
        src_actions:list,
        target_state_var:list,
        target_actions:list,
        init_pop_size:int=1000,
        crossover_rate:float=0.8,
        mutation_rate:float=0.2,
        crossover_strat:str='tournament',
        max_evol_iter:int=1000
    ):
        # save class variables
        self.src_state_var = src_state_var
        self.src_actions = src_actions
        self.target_state_var = target_state_var
        self.target_actions = target_actions
        # assign a unique code to each state variable and action for efficiency
        self.src_state_codes = {src_state_variable : id for src_state_variable, id in zip(src_state_var, range(len(src_state_var)))}
        self.src_action_codes = {src_action : id for src_action, id in zip(src_actions, range(len(src_actions)))}
        self.target_state_codes = {target_state_variable : id for target_state_variable, id in zip(target_state_var, range(len(target_state_var)))}
        self.target_action_codes = {target_action : id for target_action, id in zip(target_actions, range(len(target_actions)))}

        # evolutionary algorithm parameters
        self.init_pop_size = init_pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.crossover_strat = crossover_strat
        self.max_evol_iter = max_evol_iter

        # initialize initial population

        # evaluate initial population's fitness

        # main evolution loop
        # select parents for crossover and generate offspring

        # mutate offspring

        # evaluate offspring fitness

        # replace population with offspring

    def init_pop(self, strategy:str='random'):
        """
        Description:
            Initializes a population of inter-task mappings.

        Arguments:
            

        Return:
            (None)
        """
        population = []
        if strategy == 'random': # randomly generate mappings. Each mapping is sampled uniformly
            for _ in range(self.init_pop_size):
                # first generate the state mapping
                state_mapping_chrom = [random.randint(0, len(self.src_state_var) - 1) for _ in range(len(self.target_state_var))]
                # then generate the action mapping
                action_mapping_chrom = [random.randint(0, len(self.src_actions) - 1) for _ in range(len(self.target_actions))]
                intertask_mapping_individual = IntertaskMapping(state_mapping_chrom, action_mapping_chrom, self.src_state_var, self.src_actions, self.target_state_var, self.target_actions)
                # append individual to the population
                population.append(intertask_mapping_individual)
        # save initial population
        self.population = population

if __name__ == '__main__':
    MC2D_states = ['x_position', 'x_velocity']
    MC3D_states = ['x_position', 'y_position', 'x_velocity', 'y_velocity']
    MC2D_actions = ['Left', 'Neutral', 'Right']
    MC3D_actions = ['Neutral', 'West', 'East', 'South', 'North']

    # evolution parameters
    src_state_var = MC2D_states
    src_actions = MC2D_actions
    target_state_var = MC3D_states
    target_actions = MC3D_actions
    init_pop_size = 100
    crossover_rate = 0.8
    mutation_rate = 0.2
    crossover_strat = 'tournament'
    max_evol_iter = 1000

    ea = GAME(src_state_var, src_actions, target_state_var, target_actions, init_pop_size, crossover_rate, mutation_rate, crossover_strat, max_evol_iter)
    ea.init_pop()
    print('Done!')