# Author: Minh Hua
# Date: 10/24/2022
# Purpose: This module contains helper functions for the entire project.

import numpy as np

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