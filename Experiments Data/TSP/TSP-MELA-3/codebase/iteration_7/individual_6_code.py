import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    d_clip = np.clip(distance_matrix, 1e+5, None)
    adaptive_scale = np.median(d_clip)  # robust central tendency
    return (1/d_clip) * (1 + np.exp(-d_clip/adaptive_scale))  # stabilized inverse + local sensitivity  
    #EVOLVE-END       
    return 1 / distance_matrix