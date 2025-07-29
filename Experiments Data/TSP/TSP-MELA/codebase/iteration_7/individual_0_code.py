import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Adaptive coefficients balancing exploration/exploitation
        adaptive_weights = np.log(max(distance_matrix)+2)
        alpha = 0.5 * adaptive_weights  
        beta = 1.5 * adaptive_weights
        
        # Numerical stability system
        safe_transform = lambda x: np.sign(x)*np.log1p(np.abs(x))
        distance_matrix = distance_matrix + np.spacing(1)
        
        # Robust heuristic calculation
        log_vis = safe_transform(1/(distance_matrix))
        log_phero = safe_transform(np.ones_like(distance_matrix))
        heuristic = np.nan_to_num(
            np.exp(alpha*log_phero + beta*log_vis),
            nan=0.0, posinf=1e9, neginf=0
        )
        
        # Smart normalization
        denominator = np.sum(heuristic)
        return heuristic/(denominator if denominator else len(heuristic))
        
    except Exception:
        return np.ones_like(distance_matrix)/len(distance_matrix)
    #EVOLVE-END