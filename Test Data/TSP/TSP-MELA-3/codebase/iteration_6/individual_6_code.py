import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    d_clip = np.clip(distance_matrix, 1e-5, 1e5)  # safer clipping bounds
    mu = np.mean(np.ma.array(d_clip, mask=(d_clip==0)))  # robust mean calculation
    return np.exp(1/(d_clip + 1e-10*(d_clip<1e-5))) * (1 - np.exp(-d_clip/mu))      
    #EVOLVE-END       
    return 1 / distance_matrix