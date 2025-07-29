import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    dynamic_epsilon = 0.01 * np.mean(distance_matrix)
    dist_normalized = distance_matrix / np.max(distance_matrix)
    weights = np.exp(-2 * dist_normalized)  # More inverse component for larger distances
    hybrid = weights * np.exp(1/(distance_matrix + dynamic_epsilon)) + (1-weights) / (distance_matrix + dynamic_epsilon)
    return hybrid
    #EVOLVE-END       
    return 1 / distance_matrix