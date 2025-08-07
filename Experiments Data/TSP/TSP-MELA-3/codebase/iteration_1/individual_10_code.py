import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    alpha = 1.0  # Pheromone importance
    beta = 2.0    # Heuristic importance
    epsilon = 1e-6  # Small constant
    
    return (np.power(1/(distance_matrix + epsilon), beta)) * np.power(np.ones_like(distance_matrix), alpha)
    #EVOLVE-END       
    return 1 / distance_matrix