import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    adaptive_scale = np.percentile(distance_matrix, 75)  # More sensitive scaling
    sparsity = np.mean(distance_matrix > adaptive_scale)
    dynamic_beta = 1.2 + 0.8*sparsity  # Density-adaptive exponent
    core_heuristic = (1/(distance_matrix + epsilon))**dynamic_beta
    return core_heuristic * np.log1p(adaptive_scale/(distance_matrix + epsilon))
    #EVOLVE-END       
    return 1 / distance_matrix