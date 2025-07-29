import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Adaptive parameters
        min_dist = np.min(distance_matrix[distance_matrix > 0])
        alpha = 2.0 - np.log(min_dist + 1e-16)
        beta = 1.0 + alpha*0.5
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix")
            
        visibility = 1 / np.maximum(distance_matrix, 1e-16)
        visibility[np.isnan(visibility)] = 0.0
        
        heuristic = np.power(visibility, beta)  # Single efficient power operation
        np.fill_diagonal(heuristic, 0)  # Clear diagonal
        tot = np.sum(heuristic)
        return heuristic/tot if tot > 0 else np.ones_like(heuristic)/heuristic.size
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
    #EVOLVE-END