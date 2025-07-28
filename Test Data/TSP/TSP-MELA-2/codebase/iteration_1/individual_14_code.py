import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    scaled = np.exp(-distance_matrix / distance_matrix.mean())
    return np.where(distance_matrix > 0, scaled, 0)
    #EVOLVE-END
    return 1 / distance_matrix