import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    adaptive_exp = np.exp(1/(distance_matrix + eps))
    return (adaptive_exp / (distance_matrix + eps))
    #EVOLVE-END       
    return 1 / distance_matrix