import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    nearest_dist = np.min(distance_matrix + np.eye(distance_matrix.shape[0])*1e6, axis=1)
    neighb_scale = np.mean(np.sort(distance_matrix, axis=1)[:, 1:2], axis=1)
    exp_decay =np.exp(-distance_matrix/(nearest_dist[:,None] * neighb_scale[:,None] + eps))
    return exp_decay/(distance_matrix + eps*(distance_matrix==0))
    #EVOLVE-END
    return 1 / distance_matrix