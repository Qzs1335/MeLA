import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-5
    log_scaled = np.log1p(distance_matrix) + eps
    adaptive_weight = 0.5 + np.tanh(distance_matrix - distance_matrix.mean())
    exp_scaling = np.exp(adaptive_weight/(distance_matrix+eps))
    return (1 - exp_scaling) / (distance_matrix * log_scaled)
    #EVOLVE-END
    return 1 / distance_matrix