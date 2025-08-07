import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-15
    blended = 0.99/(distance_matrix**2 + epsilon) + 0.01/distance_matrix
    return blended
    #EVOLVE-END       
    return 1 / distance_matrix