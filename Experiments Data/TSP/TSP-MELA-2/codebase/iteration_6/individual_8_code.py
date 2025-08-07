import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-6
    neigh_weights = np.mean(np.sort(distance_matrix, axis=1)[:,1:5], axis=1)
    exp_decay = np.exp(-(distance_matrix**2)/neigh_weights[:,None])
    return exp_decay / (distance_matrix + eps)
    #EVOLVE-END       
    return 1 / distance_matrix