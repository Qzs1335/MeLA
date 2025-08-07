import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    exp_decay = np.exp(-distance_matrix / np.mean(distance_matrix))
    return exp_decay / (np.sqrt(distance_matrix) + eps)
    #EVOLVE-END
    return 1 / distance_matrix