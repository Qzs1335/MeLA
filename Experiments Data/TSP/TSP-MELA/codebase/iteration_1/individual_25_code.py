import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    decay_factor = 0.5
    base_heuristic = 1 / (distance_matrix + 1e-10)
    weighted = base_heuristic * np.exp(-decay_factor * distance_matrix)
    return weighted / (weighted.sum(axis=1, keepdims=True) + 1e-10)
    #EVOLVE-END
    return 1 / distance_matrix