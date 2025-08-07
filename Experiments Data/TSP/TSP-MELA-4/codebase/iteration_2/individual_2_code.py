import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5
    alpha = 1.0
    beta = 2.0
    normalized_dist = distance_matrix / np.max(distance_matrix)
    return alpha * np.exp(-normalized_dist/5) * ((1/(distance_matrix + epsilon))**beta)
    #EVOLVE-END