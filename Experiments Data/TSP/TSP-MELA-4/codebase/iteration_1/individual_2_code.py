import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10  # Small constant to avoid division by zero
    return np.exp(-distance_matrix/10) + epsilon
    #EVOLVE-END       
    return 1 / distance_matrix