import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-12  # Enhanced num stability
    adaptive_scale = np.median(distance_matrix[distance_matrix>0])
    log_scaled = np.log1p(distance_matrix/adaptive_scale)
    exp_term = np.exp(-distance_matrix/(adaptive_scale + eps))
    return exp_term/(distance_matrix + eps) * log_scaled
    #EVOLVE-END       
    return 1 / distance_matrix