import numpy as np
import numpy as np  
def heuristics_v2(distance_matrix):
    #EVOLVE-START  
    epsilon = 1e-10  
    inv_dist = 1/(distance_matrix + epsilon)  
    scaled = np.exp(0.5 * inv_dist)  # Balanced scaling  
    return scaled / np.max(scaled)   # Normalized output  
    #EVOLVE-END       
    return 1 / distance_matrix