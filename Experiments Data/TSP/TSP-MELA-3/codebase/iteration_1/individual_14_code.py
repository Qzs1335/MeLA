import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    distance_weight = np.sqrt(distance_matrix)
    log_adjustment = np.log(distance_matrix + 1)
    combined_heuristic = (1 / distance_weight) * log_adjustment
    return combined_heuristic
    #EVOLVE-END       
    return 1 / distance_matrix