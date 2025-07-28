import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    scale = np.mean(distance_matrix)
    decay = np.exp(-distance_matrix/(scale + epsilon))
    return decay / (distance_matrix + epsilon)**1.5
    #EVOLVE-END       
    return 1 / distance_matrix