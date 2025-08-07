import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    neigh_dist = np.sort(distance_matrix, axis=1)[:,1:5]
    neigh_weights = np.median(neigh_dist, axis=1, keepdims=True)
    sigmoid = 1/(1+np.exp(-distance_matrix/neigh_weights))
    norm_neigh = (neigh_dist/np.sum(neigh_dist, axis=1, keepdims=True))[:,0:1]
    return sigmoid * (2/(1/distance_matrix + 1/neigh_weights)) * norm_neigh
    #EVOLVE-END
    return 1 / distance_matrix