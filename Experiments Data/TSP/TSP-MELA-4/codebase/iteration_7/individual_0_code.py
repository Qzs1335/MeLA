import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    adaptive_scale = np.median(distance_matrix)
    dynamic_beta = 1.0 + distance_matrix/adaptive_scale  # linear adaptation
    sigmoid_decay = np.exp(-distance_matrix/(adaptive_scale + epsilon))/(1+np.exp(-distance_matrix))
    return (1/(distance_matrix + epsilon))**dynamic_beta * np.sqrt(sigmoid_decay)
    #EVOLVE-END       
    return 1 / distance_matrix