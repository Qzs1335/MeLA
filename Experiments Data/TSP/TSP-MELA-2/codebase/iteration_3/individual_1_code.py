import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    k = max(2, int(np.log2(distance_matrix.shape[0])))  # Adaptive neighborhood size
    nearest_neigh = np.sort(distance_matrix, axis=1)[:,1:k+1]
    neigh_weights = np.median(nearest_neigh, axis=1)
    decay = 1/(1 + distance_matrix/neigh_weights[:,None])
    return decay*(decay + 1.5)/(distance_matrix + eps)  # Smoother transition
    #EVOLVE-END       
    return 1/distance_matrix