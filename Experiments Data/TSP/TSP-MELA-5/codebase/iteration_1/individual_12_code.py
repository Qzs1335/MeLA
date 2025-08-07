import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    min_span = np.min(distance_matrix, axis=1, keepdims=True)
    log_dist = np.log1p(distance_matrix)
    softmax = np.exp(-log_dist) / np.sum(np.exp(-log_dist), axis=1, keepdims=True)
    return (1/distance_matrix) * softmax * min_span
    #EVOLVE-END       
    return 1 / distance_matrix