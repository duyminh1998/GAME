# Author: Minh Hua
# Date: 11/1/2022
# Purpose: This module contains the main evolutionary algorithm that will be used to evolve inter-task mappings.

class IntertaskMapping:
    """A class that represents an inter-task mapping consisting of a state mapping and action mapping."""
    def __init__(self, state_mapping:list, action_mapping:list, target) -> None:
        self.state_mapping = state_mapping
        self.action_mapping = action_mapping

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

    def init_pop(self):
        """
        Description:
            Initializes a population of inter-task mappings.

        Arguments:
            

        Return:
            (None)
        """