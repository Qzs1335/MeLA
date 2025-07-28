import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    # Clip distances to avoid extreme values
    safe_dist = np.clip(distance_matrix, epsilon, 1/epsilon)
    
    # Compute terms with numerical stability
    log_term = -np.log(safe_dist)
    recip_term = 1/safe_dist
    
    # Compute geometric mean with stability
    hybrid = np.sqrt(log_term * recip_term + epsilon)
    
    # Normalize and handle potential NaN/inf
    norm = np.sum(hybrid, axis=1, keepdims=True)
    norm[norm == 0] = 1  # Avoid division by zero
    result = hybrid / norm
    
    # Final sanity check and cleanup
    result = np.nan_to_num(result, nan=1/result.shape[1], posinf=1/result.shape[1], neginf=1/result.shape[1])
    result = np.clip(result, 0, 1)  # Ensure valid probability range
    
    # Renormalize after cleanup
    norm = np.sum(result, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return result / norm
    #EVOLVE-END
