import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    d_clip = np.clip(distance_matrix, 1e-5, None)
    scale = 1 / (1 + np.exp(-0.5*distance_matrix/np.mean(distance_matrix)))  # sigmoid scaling
    return 1 / (d_clip ** (0.5 + scale))  # power adapts with distance
    #EVOLVE-END       
    return 1 / distance_matrix