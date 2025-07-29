import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    dist_log = np.log(distance_matrix + epsilon)
    return (1 / (distance_matrix + epsilon)) * (1 + 0.5 * dist_log)
    #EVOLVE-END
    return 1 / distance_matrix