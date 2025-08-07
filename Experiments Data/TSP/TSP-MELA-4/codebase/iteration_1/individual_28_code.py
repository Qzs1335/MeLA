import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    log_scale = np.log1p(distance_matrix)
    exp_decay = np.exp(-distance_matrix/10)
    return (0.6/distance_matrix + 0.2/log_scale + 0.2*exp_decay) + epsilon
    #EVOLVE-END
    return 1 / distance_matrix