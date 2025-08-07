import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5
    scaled = np.exp(-distance_matrix/10)
    return np.log(1/(distance_matrix + epsilon) + scaled)
    #EVOLVE-END       
    return 1 / distance_matrix