import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-08  # Adjusted for better gradient balance
    weights = 0.3/distance_matrix + 0.7*np.exp(1/(distance_matrix + epsilon))
    return weights
    #EVOLVE-END       
    return 1 / distance_matrix