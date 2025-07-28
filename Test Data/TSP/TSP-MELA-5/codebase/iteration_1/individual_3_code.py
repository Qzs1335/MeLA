import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    min_dist = np.min(distance_matrix[distance_matrix > 0])
    normalized_dist = distance_matrix / min_dist
    return np.exp(-0.5 * np.log1p(normalized_dist))
    #EVOLVE-END
    return 1 / distance_matrix