import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    adaptive_scale = np.median(distance_matrix)
    dynamic_beta = 1.5 + 0.5*np.sin(adaptive_scale)
    return (1/(distance_matrix + epsilon))**dynamic_beta / adaptive_scale
    #EVOLVE-END       
    return 1 / distance_matrix