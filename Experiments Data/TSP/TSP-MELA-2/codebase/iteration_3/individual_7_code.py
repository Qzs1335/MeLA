import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-6
    median_dist = np.median(np.partition(distance_matrix,3)[:,:4])
    norm_dist = distance_matrix/(median_dist + eps)
    return np.exp(-np.abs(np.log(norm_dist + 1))) * median_dist/(distance_matrix + eps)
    #EVOLVE-END
    return 1 / distance_matrix