import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    safe_dist = distance_matrix + eps
    neigh_weights = np.median(np.sort(safe_dist,axis=1)[:,1:5], axis=1)
    scaled = np.log1p(safe_dist)
    decay = np.exp(-safe_dist/(2*neigh_weights[:,None]+eps))
    return decay * neigh_weights[:,None] / scaled
    #EVOLVE-END
    return 1 / distance_matrix