import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-6
    smoothing = np.exp(-distance_matrix/np.median(distance_matrix))
    return smoothing / (distance_matrix + epsilon)
    #EVOLVE-END       
    return 1 / distance_matrix