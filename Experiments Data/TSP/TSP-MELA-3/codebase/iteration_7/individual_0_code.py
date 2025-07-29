import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    d_norm = np.log1p(distance_matrix) + 1e-16
    sig_weight = 1/(1+np.exp(d_norm-np.median(d_norm)))
    return sig_weight*np.exp(-d_norm) + (1-sig_weight)/d_norm
    #EVOLVE-END       
    return 1 / distance_matrix