import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    q25 = np.quantile(distance_matrix, 0.25, axis=1)
    norms = (distance_matrix - q25[:,None]) / (np.max(distance_matrix, axis=1) - q25 + eps)[:,None]
    return np.exp(-norms) / (distance_matrix + eps)**0.5
    #EVOLVE-END       
    return 1 / distance_matrix