import numpy as np
import numpy as np  

def heuristics_v2(distance_matrix):  
    #EVOLVE-START  
    try:  
        alpha = np.clip(0.5 + np.random.rand(), 0.5, 2.5)  # dynamic weight  
        beta = np.clip(3.0 - alpha, 0.5, 3.0)             # complementary weight  
        eps = np.maximum(1e-16, np.min(distance_matrix[distance_matrix > 0])/1e8)  

        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)  
        valid_mask = (distance_matrix > 0) & ~np.eye(distance_matrix.shape[0], dtype=bool)  
        visibility = np.where(valid_mask, 1/np.clip(distance_matrix, eps, None), 0)  

        log_terms = alpha * np.log(np.ones_like(distance_matrix) + eps) + \  
                   beta * np.log(visibility + eps)  
        heuristic = np.exp(np.clip(log_terms, -100, 100))  

        return np.exp(heuristic) / np.sum(np.exp(heuristic)) if heuristic.size > 0 else \  
               np.ones_like(distance_matrix)/distance_matrix.size  

    except Exception as e:  
        print(f"Heuristic Error: {str(e)}")  
        return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None  
    #EVOLVE-END