import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    epsilon = 1e-10
    log_d = np.log(np.clip(distance_matrix, 1e-5, None))
    sig_weights = 1/(1+np.exp(log_d-np.median(log_d)))  # sigmoid scaling
    hybrid = sig_weights*np.exp(-log_d) + (1-sig_weights)*np.exp(log_d)
    return np.exp(-hybrid)  # final normalization
    #EVOLVE-END       
    return 1 / distance_matrix