# Author: Minh Hua
# Date: 10/24/2022
# Purpose: This module contains helper functions for the entire project.

from collections import namedtuple
import pandas as pd

# Named tuple that stores metadata about an experiment
ExperimentInfo = namedtuple('ExperimentInfo', [
    'env_name', 'env_max_steps', 'env_seed',
    'env_max_episodes', 'agent_name'
])

# Named tuple that stores metadata about an agent
SarsaLambdaAgentInfo = namedtuple('SarsaLambdaAgentInfo', [
    'alpha', 'lamb', 'gamma', 'method',
    'epsilon', 'num_of_tilings', 'max_size'
])

class RLSamplesCollector:
    """This class collects samples from RL training episodes and saves them to a csv"""
    def __init__(
        self,
        experiment_info:ExperimentInfo,
        agent_info:SarsaLambdaAgentInfo,
        data_column_names:list,
        data_column_dtypes:list
    ) -> None:
        """
        Description:
            Initializes a Sarsa(lambda) agent using CMAC tile coding.

        Arguments:
            experiment_info: metadata about the experiment.
            agent_info: metadata about the agent.
            data_column_names: a list of column names for the data.
            data_column_dtypes: a list of dtypes for the columns.

        Return:
            (None)
        """
        # save metadata
        self.experiment_info = experiment_info
        self.agent_info = agent_info

        # build pandas DataFrame using data_column_names
        data_column_info = {col_name : col_dtype for col_name, col_dtype in zip(data_column_names, data_column_dtypes)}
        self.data = pd.DataFrame({c: pd.Series(dtype=dt) for c, dt in data_column_info.items()})

    def log_data(self, data_pt:dict) -> None:
        """
        Description:
            Appends a data point to the dataframe.

        Arguments:
            data_pt: a dictionary where the keys are the column names and the items are the data values.

        Return:
            (None)
        """
        self.data = self.data.append(data_pt, ignore_index=True)

    def export_data(self, path:str, file_name:str) -> None:
        """
        Description:
            Saves the dataframe to an external file.

        Arguments:
            path: the path to save the dataframe.
            file_name: the name for the file.

        Return:
            (None)
        """
        self.data.to_csv(path + file_name, index = False)

    def write_metadata(self, path:str, file_name:str) -> None:
        """
        Description:
            Saves the metadata to an external file.

        Arguments:
            path: the path to save the metadata.
            file_name: the name for the file.

        Return:
            (None)
        """
        with open(path + file_name, 'w') as f:
            sep = '-------------------------------------------------------------'
            f.write('Experiment Info \n' + sep + '\n')
            for k, v in self.experiment_info._asdict().items():
                f.write('{} = {} \n'.format(k, v))
            f.write('\nAgent Info \n' + sep + '\n')
            for k, v in self.agent_info._asdict().items():
                f.write('{} = {} \n'.format(k, v))