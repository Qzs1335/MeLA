import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    nn_heuristic = np.partition(distance_matrix, 1, axis=1)[:,1:4].mean(axis=1)
    scaled_dist = distance_matrix / distance_matrix.mean()
    return 0.7 * (1/distance_matrix) + 0.3 * (1/nn_heuristic[:,None]) * np.log(1 + scaled_dist)
    #EVOLVE-END
    return 1 / distance_matrix