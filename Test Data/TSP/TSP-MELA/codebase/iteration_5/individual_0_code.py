import numpy as np
import numpy as np  

def heuristics_v2(distance_matrix):
    #EVOLVE-START  
    try:  
        # Dynamic parameters  
        beta = max(1.5, np.log1p(distance_matrix.mean()))  
        decay = 0.25 + 0.75/(1 + distance_matrix.shape[0]**0.5)  
        epsilon = np.finfo(float).eps * 100  

        np.fill_diagonal(distance_matrix, np.inf)  
        visible = 1 / np.maximum(distance_matrix, epsilon)  
        
        # Enhanced log-transform  
        log_heap = decay * np.log1p(visible*beta)  
        heuristic = np.exp(log_heap - log_heap.max())  
        
        # Robust normalization  
        softmax = np.clip(heuristic, 1e-16, 1).ravel()  
        return (softmax/softmax.sum()).reshape(visible.shape)  

    except Exception:  
        fallback = np.ones_like(distance_matrix)  
        return fallback/fallback.size  
    #EVOLVE-END