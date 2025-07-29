import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    base_heuristic = 1 / distance_matrix
    decay_factor = np.exp(-distance_matrix/np.mean(distance_matrix))
    log_scaled = np.log(1 + base_heuristic)
    return base_heuristic * decay_factor * log_scaled
    #EVOLVE-END       
    return 1 / distance_matrix