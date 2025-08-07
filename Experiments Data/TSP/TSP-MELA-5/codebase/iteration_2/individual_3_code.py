import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    smoothed = 1 / (distance_matrix + epsilon)
    return np.log1p(smoothed) * np.exp(-distance_matrix/10)
    #EVOLVE-END       
    return 1 / distance_matrix