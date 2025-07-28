import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    log_scale = np.log1p(distance_matrix)
    return (1 / (distance_matrix + epsilon)) * (1 / (log_scale + epsilon))
    #EVOLVE-END
    return 1 / distance_matrix