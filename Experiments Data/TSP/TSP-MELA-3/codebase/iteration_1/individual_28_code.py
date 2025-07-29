import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10  # small constant to avoid division by zero
    distance_matrix = np.maximum(distance_matrix, epsilon)  # ensure non-zero distances
    normalized = distance_matrix / np.max(distance_matrix)
    return np.exp(-3 * normalized)  # exponential decay heuristic
    #EVOLVE-END       
    return 1 / distance_matrix