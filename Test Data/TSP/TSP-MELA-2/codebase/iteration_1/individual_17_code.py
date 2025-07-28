import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    ranked = np.argsort(distance_matrix, axis=1)
    rank_weights = 1 / (1 + np.arange(ranked.shape[1]))
    visibility = rank_weights[ranked] 
    return np.log(visibility + 1e-10) / (distance_matrix + 1e-10)
    #EVOLVE-END       
    return 1 / distance_matrix