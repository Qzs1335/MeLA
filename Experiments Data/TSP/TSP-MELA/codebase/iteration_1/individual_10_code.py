import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    visibility = 1 / (distance_matrix + 1e-10)  # Smoothing added
    adaptive_exponent = np.clip(1.0 + np.random.normal(0, 0.1), 0.8, 1.2)
    heuristics = np.power(visibility, adaptive_exponent)
    #EVOLVE-END     
    return heuristics