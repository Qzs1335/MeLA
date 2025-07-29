import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    scale = np.median(distance_matrix)/10
    softplus = np.log1p(np.exp(distance_matrix/scale))
    return (1 / (distance_matrix + scale)) * (1 / softplus)
    #EVOLVE-END       
    return 1 / distance_matrix