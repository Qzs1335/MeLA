import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    ε = 1e-10  # small constant to avoid division by zero
    return np.log(1 + (1 / (distance_matrix + ε)))
    #EVOLVE-END  
    return 1 / distance_matrix