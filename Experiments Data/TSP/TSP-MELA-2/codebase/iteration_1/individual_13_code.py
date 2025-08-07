import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    centrality = np.sum(1/distance_matrix, axis=1)
    scaled_dist = np.log1p(distance_matrix)
    combined = centrality.reshape(-1,1) * (1/scaled_dist)
    return combined / np.max(combined)
    #EVOLVE-END       
    return 1 / distance_matrix