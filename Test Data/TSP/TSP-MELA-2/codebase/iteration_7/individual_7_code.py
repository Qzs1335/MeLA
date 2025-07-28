import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    # EVOLVE-START
    eps = 1e-8
    decay = np.exp(-np.sqrt(distance_matrix))
    return (decay + 1) / (distance_matrix + eps)
    # EVOLVE-END
    return 1 / distance_matrix