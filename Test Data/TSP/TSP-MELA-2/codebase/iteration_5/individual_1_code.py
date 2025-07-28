import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    noise = (1e-3 * np.random.rand(*distance_matrix.shape))  
    norm_dist = distance_matrix / (np.mean(distance_matrix) + eps)  
    tanh_grad = 1 - np.tanh(norm_dist)  
    log_scaled = np.log(norm_dist + 1)  
    return (tanh_grad * log_scaled) / (distance_matrix**2 + eps) + noise
    #EVOLVE-END
    return 1 / distance_matrix