import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    return np.log(1 + (1 / (distance_matrix + epsilon)**0.7))
    #EVOLVE-END
    return 1 / distance_matrix