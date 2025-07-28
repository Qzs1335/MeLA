import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-6
    distance_variance = np.var(distance_matrix, axis=1, keepdims=True)
    normalized_dist = (distance_matrix - np.min(distance_matrix)) / (np.max(distance_matrix) - np.min(distance_matrix) + eps)
    return (1/(distance_matrix + eps)) * (1 + distance_variance) * normalized_dist
    #EVOLVE-END
    return 1 / distance_matrix