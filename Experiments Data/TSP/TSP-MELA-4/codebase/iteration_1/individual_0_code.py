import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    log_scaled = np.log(1 + distance_matrix)
    return np.exp(-0.5 * (distance_matrix**2)) / (log_scaled + epsilon)
    #EVOLVE-END
    return 1 / distance_matrix