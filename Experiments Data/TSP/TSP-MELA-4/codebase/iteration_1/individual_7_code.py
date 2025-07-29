import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    log_scaled = np.log(distance_matrix + epsilon)
    return np.exp(-0.5 * log_scaled) / (distance_matrix + epsilon)
    #EVOLVE-END
    return 1 / distance_matrix