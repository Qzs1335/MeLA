import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    log_scaled = np.log10(distance_matrix + 1 + eps)
    exp_decay = np.exp(-distance_matrix/np.mean(distance_matrix, axis=0)[None,:])
    return exp_decay / log_scaled
    #EVOLVE-END       
    return 1 / distance_matrix