import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    epsilon = 1e-10
    d_clip = np.clip(distance_matrix + epsilon, 1e-5, None)
    density = np.exp(-distance_matrix/d_clip.mean())
    return (density * np.exp(1/d_clip) + (1-density)/(d_clip+epsilon))        
    #EVOLVE-END       
    return 1 / distance_matrix