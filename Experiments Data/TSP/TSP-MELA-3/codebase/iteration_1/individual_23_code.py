import numpy as np
import numpy as np  
def heuristics_v2(distance_matrix):  
    #EVOLVE-START  
    epsilon = 1e-5  # prevent division by zero  
    scaled = np.log(distance_matrix + 1)  
    return 1 / (np.sqrt(distance_matrix) + epsilon) * np.exp(-2*scaled)  
    #EVOLVE-END       
    return 1 / distance_matrix