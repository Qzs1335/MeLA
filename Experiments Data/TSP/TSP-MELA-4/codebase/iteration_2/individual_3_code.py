import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5
    beta = 2.0
    normalized_dist = distance_matrix/np.max(distance_matrix)
    return (1/(normalized_dist + epsilon))**beta * np.exp(-normalized_dist)
    #EVOLVE-END
    return 1 / distance_matrix