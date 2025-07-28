import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    epsilon = 1e-10
    d_norm = distance_matrix/(np.median(distance_matrix)+epsilon)
    d_clip = np.clip(d_norm, 1e-5, 1e5)  # balanced clipping
    weights = 0.5*(1 + np.tanh(1-d_clip))  # smoother weighting
    hybrid = weights*np.exp(1/d_clip) + (1-weights)/d_clip
    return hybrid        
    #EVOLVE-END       
    return 1 / distance_matrix