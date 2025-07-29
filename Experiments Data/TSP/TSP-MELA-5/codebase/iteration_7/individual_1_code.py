import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-15
    softplus_weights = np.log1p(np.exp(distance_matrix)) 
    return np.sqrt(1/(distance_matrix + epsilon)) * (1/softplus_weights)
    #EVOLVE-END       
    return 1 / distance_matrix