import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    ranked_distances = np.argsort(np.argsort(distance_matrix, axis=1), axis=1) + 1
    scaled_weights = np.exp(-0.1 * ranked_distances)
    smooth_factor = 1/np.mean(distance_matrix)
    return scaled_weights/(distance_matrix + np.percentile(distance_matrix,10)*smooth_factor)
    #EVOLVE-END       
    return 1 / distance_matrix