import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    power = 1.5
    return np.exp(-0.5 * (distance_matrix**power)) / (distance_matrix + epsilon)
    #EVOLVE-END       
    return 1 / distance_matrix