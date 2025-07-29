import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    log_term = np.log1p(distance_matrix)
    nearest_neigh = np.sort(distance_matrix,axis=1)[:,1:4]
    median_dist = np.median(nearest_neigh,axis=1)
    
    linear_decay = distance_matrix/(median_dist[:,None]+eps)
    dynamic_scale = (distance_matrix - np.min(distance_matrix))/(np.max(distance_matrix)-np.min(distance_matrix)+eps)
    return (1/(distance_matrix+eps)) * np.exp(-linear_decay) * dynamic_scale
    #EVOLVE-END       
    return 1 / distance_matrix