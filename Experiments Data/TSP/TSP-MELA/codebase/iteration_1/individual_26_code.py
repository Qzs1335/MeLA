import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5
    log_term = np.log1p(distance_matrix)
    power_term = np.power(distance_matrix, 0.3)
    return (1/(distance_matrix+epsilon)) * (1/power_term) * log_term
    #EVOLVE-END       
    return 1 / distance_matrix