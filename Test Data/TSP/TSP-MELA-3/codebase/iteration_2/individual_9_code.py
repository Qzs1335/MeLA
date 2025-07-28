import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    max_dist = np.max(distance_matrix, axis=1, keepdims=True)
    normalized = distance_matrix / (max_dist + epsilon)
    return 0.5 * (1/(normalized + epsilon) + np.exp(-normalized))
    #EVOLVE-END       
    return 1 / distance_matrix