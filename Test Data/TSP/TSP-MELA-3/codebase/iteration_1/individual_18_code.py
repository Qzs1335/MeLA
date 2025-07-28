import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    noise = np.random.rand(*distance_matrix.shape) * 0.1
    return (1 / distance_matrix) * (1 + noise)
    #EVOLVE-END       
    return 1 / distance_matrix