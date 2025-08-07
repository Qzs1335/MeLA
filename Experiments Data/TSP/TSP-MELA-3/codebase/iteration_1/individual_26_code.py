import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    avg_distances = np.mean(distance_matrix, axis=1)
    normalized_dist = distance_matrix / np.max(distance_matrix)
    return np.exp(-normalized_dist) * (1 / (1 + avg_distances[:, np.newaxis]))
    #EVOLVE-END
    return 1 / distance_matrix