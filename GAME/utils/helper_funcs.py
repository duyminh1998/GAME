# Author: Minh Hua
# Date: 10/24/2022
# Purpose: This module contains helper functions for the entire project.

import numpy as np
import pickle

def distance(p1:tuple, p2:tuple) -> float:
    """
    Description:
        Returns the Euclidean distance between two points.

    Arguments:
        p1: the first point.
        p2: the second point.

    Return:
        (float) the Euclidean distance between two points.
    """
    return np.sum(np.square(np.array(p1) - np.array(p2)))

def angle(a:tuple, b:tuple, c:tuple) -> float:
    """
    Description:
        Returns the angle between two vectors, as defined by three points.

    Arguments:
        a: the first point.
        b: the second point. Usually the middle point that the two vectors share.
        c: the third point.

    Return:
        (float) the angle in degrees between the three poitns.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def save_agent_to_file(agent, path:str, agent_filename:str, agent_weights_filename:bool=True, agent_hash_filename:bool=True, agent_z_filename:bool=True) -> None:
    """
    Description:
        Saves a Sarsa lambda agent.

    Arguments:
        agent: the agent to be saved.
        path: the path containing the folder to save agent data.
        agent_filename: the filename for the agent.
        agent_weights_filename: the filename for the agent's weights.
        agent_hash_filename: the filename for the agent's hash table.
        agent_z_filename: the filename for the agent's trace.

    Return:
        (None) 
    """
    with open(path + agent_filename, 'wb') as f:
        pickle.dump(agent, f)
    if agent_weights_filename:
        agent_weights_filename = agent_filename.split('.')[1] + '_weights.pickle'
        with open(path + agent_weights_filename, 'wb') as f:
            pickle.dump(agent.weights, f)
    if agent_hash_filename:
        agent_hash_filename = agent_filename.split('.')[1] + '_hash.pickle'
        with open(path + agent_hash_filename, 'wb') as f:
            pickle.dump(agent.hash_table, f)
    if agent_z_filename:
        agent_z_filename = agent_filename.split('.')[1] + '_z.pickle'
        with open(path + agent_z_filename, 'wb') as f:
            pickle.dump(agent.z, f)