import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    adaptive_scale = np.median(distance_matrix)
    dynamic_beta = 1.0 + 0.5*np.sin(adaptive_scale/2)  # smoother adaptation
    return (1/(distance_matrix + epsilon))**dynamic_beta
    #EVOLVE-END       
    return 1 / distance_matrix