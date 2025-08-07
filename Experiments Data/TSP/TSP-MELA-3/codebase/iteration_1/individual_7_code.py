import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    scaled = np.exp(-distance_matrix / distance_matrix.min())
    clipped = np.clip(scaled, 1e-5, 1e5)
    return clipped
    #EVOLVE-END       
    return 1 / distance_matrix