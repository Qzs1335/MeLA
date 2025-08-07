import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    neigh_mean = np.mean(np.sort(distance_matrix, axis=1)[:,1:4], axis=1)
    hybrid_decay = (np.exp(-distance_matrix/neigh_mean[:,None]) + 1/(distance_matrix+eps))
    return hybrid_decay * neigh_mean[:,None] / np.sqrt(distance_matrix + eps)
    #EVOLVE-END
    return 1 / distance_matrix