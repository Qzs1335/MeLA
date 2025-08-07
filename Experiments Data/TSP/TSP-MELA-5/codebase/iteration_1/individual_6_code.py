import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    log_scaled = np.log1p(distance_matrix)
    decay_factor = np.exp(-0.5 * distance_matrix / distance_matrix.mean())
    heuristic = decay_factor / log_scaled
    return heuristic / heuristic.max()
    #EVOLVE-END
    return 1 / distance_matrix