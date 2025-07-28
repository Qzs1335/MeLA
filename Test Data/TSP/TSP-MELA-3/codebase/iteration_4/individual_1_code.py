import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-16
    inv = 1 / (distance_matrix + epsilon)
    exp = np.exp(inv)
    return np.exp(np.sqrt(inv * np.log(exp + 1)))  
    #EVOLVE-END       
    return 1 / distance_matrix