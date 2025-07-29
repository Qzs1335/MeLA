import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    log_term = np.log1p(distance_matrix) + 1  # Smoother gradient
    exp_decay = np.exp(-distance_matrix/np.mean(distance_matrix))  # Global scaling
    return exp_decay / (distance_matrix * log_term + eps)  # Compact formulation
    #EVOLVE-END       
    return 1 / distance_matrix