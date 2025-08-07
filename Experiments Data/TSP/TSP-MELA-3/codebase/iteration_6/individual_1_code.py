import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    epsilon = 1e-10
    d_clip = np.maximum(distance_matrix, epsilon)  # simpler stable clipping
    return 1 / d_clip  # pure inverse with ensured stability
    #EVOLVE-END       
    return 1 / distance_matrix