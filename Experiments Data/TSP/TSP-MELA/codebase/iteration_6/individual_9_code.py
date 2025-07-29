import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Dynamic parameters
        alpha = max(1, 1.5 - np.log1p(distance_matrix.shape[0])/10) 
        beta = min(3, 2 + np.log1p(distance_matrix.shape[0])/5)
        stability_factor = np.finfo(float).eps * 100
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        
        # Matrix sanitization
        safe_dist = distance_matrix.copy()
        np.fill_diagonal(safe_dist, np.inf)
        safe_dist[safe_dist <= 0] = stability_factor
        
        # Robust visibility calculation
        visibility = np.where(safe_dist > 0, 1/safe_dist, 0)
        visibility = np.nan_to_num(visibility, nan=0.0, posinf=0.0)
        
        # Adaptive transformation
        log_terms = alpha*np.log(safe_dist) - beta*np.log(visibility + stability_factor)
        heuristic = np.exp(-log_terms + stability_factor)
        
        # Softmax normalization
        exp_heu = np.exp(heuristic - np.max(heuristic))
        return exp_heu/(exp_heu.sum() + stability_factor)
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        shape = distance_matrix.shape if hasattr(distance_matrix, 'shape') else (1,)
        return np.ones(shape)/np.prod(shape)
    #EVOLVE-END