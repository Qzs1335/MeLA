import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    exp_factor = np.exp(-distance_matrix/np.mean(distance_matrix))
    log_factor = np.log(1 + 1/distance_matrix)
    return 0.7*exp_factor + 0.3*log_factor
    #EVOLVE-END
    return 1 / distance_matrix