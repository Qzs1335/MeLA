import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = np.percentile(distance_matrix, 10) * 1e-6  # Adaptive stability threshold
    proximity = 1 / (distance_matrix**2 + epsilon)
    smoothness = 1 / np.log1p(distance_matrix)  # Softer than log()
    return proximity * np.sqrt(smoothness)      # Balanced composition
    #EVOLVE-END       
    return 1 / distance_matrix