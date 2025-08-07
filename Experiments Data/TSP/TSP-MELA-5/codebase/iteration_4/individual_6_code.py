import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-15
    return (0.7/(distance_matrix + epsilon)) + (0.3/(distance_matrix**2 + epsilon))
    #EVOLVE-END       
    return 1 / distance_matrix