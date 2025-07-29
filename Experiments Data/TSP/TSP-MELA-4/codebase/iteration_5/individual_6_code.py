import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    scale = np.mean(distance_matrix)
    return (1/(distance_matrix + epsilon))**1.5 * np.log1p(1/(distance_matrix + epsilon))
    #EVOLVE-END       
    return 1 / distance_matrix