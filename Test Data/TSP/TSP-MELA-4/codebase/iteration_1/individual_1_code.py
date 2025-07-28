import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    nearest_neighbor = np.min(distance_matrix, axis=1, keepdims=True)
    return np.exp(-distance_matrix / nearest_neighbor)
    #EVOLVE-END
    return 1 / distance_matrix