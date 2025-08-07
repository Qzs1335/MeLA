import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5
    η = np.random.uniform(0.99,1.01)  # small noise
    adaptive_scale = np.mean(distance_matrix)  
    β = 1.5 + 0.5*np.sin(np.pi*np.mean(distance_matrix)/adaptive_scale)  # dynamic β
    decay = np.exp(-distance_matrix/(adaptive_scale + epsilon))
    inv_dist = (1/(distance_matrix + epsilon))**β
    return η * decay * inv_dist
    #EVOLVE-END       
    return 1 / distance_matrix