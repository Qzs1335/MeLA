import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    log_term = np.log1p(distance_matrix)
    exp_term = np.exp(-distance_matrix/np.mean(distance_matrix))
    return (exp_term + 0.5) / (log_term + epsilon)
    #EVOLVE-END
    return 1 / distance_matrix