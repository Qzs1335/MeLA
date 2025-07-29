import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    base = 1/(distance_matrix + epsilon)
    return np.tanh(np.sqrt(base)) * base
    #EVOLVE-END       
    return 1 / distance_matrix