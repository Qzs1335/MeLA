import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    d_log = np.log1p(distance_matrix)
    mst_factor = np.min(distance_matrix, axis=1, keepdims=True)
    t = 0.7
    return t*(1/(d_log+1e-10)) + (1-t)*(mst_factor/distance_matrix)
    #EVOLVE-END       
    return 1 / distance_matrix