import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-10
    log_scaled = np.log(1 + distance_matrix)
    return (1 / (distance_matrix + eps)) * (1 / log_scaled)
    #EVOLVE-END       
    return 1 / distance_matrix