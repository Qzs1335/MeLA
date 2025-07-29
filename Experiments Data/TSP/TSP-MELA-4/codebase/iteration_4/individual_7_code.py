import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5
    adaptive_scale = np.mean(distance_matrix[distance_matrix > epsilon])  # exclude near-zero distances
    decay = np.exp(-distance_matrix/(adaptive_scale + epsilon))
    inv_dist = (1/(distance_matrix + epsilon))**1.5  # tuned exponent
    return (0.6*decay + 0.4*inv_dist)  # weighted hybrid
    #EVOLVE-END       
    return 1 / distance_matrix