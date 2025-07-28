import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    distance_matrix = np.where(distance_matrix == 0, epsilon, distance_matrix)
    return np.exp(-0.5 * distance_matrix) * (1 / np.log1p(distance_matrix))
    #EVOLVE-END
    return 1 / distance_matrix