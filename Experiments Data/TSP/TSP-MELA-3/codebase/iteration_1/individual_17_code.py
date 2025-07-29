import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    normalized = np.log1p(distance_matrix)/np.max(distance_matrix)
    return np.exp(-3*normalized)
    #EVOLVE-END
    return 1 / distance_matrix