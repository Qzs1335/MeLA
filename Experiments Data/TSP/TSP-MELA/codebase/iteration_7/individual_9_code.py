import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Adaptive parameters
        beta = 2.5 - 0.5 * np.log1p(distance_matrix.mean()) if distance_matrix.size>0 else 2
        alpha = max(1.0, 3.5-beta)
        stability_factor = 1e-12
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            return np.ones(distance_matrix.shape)/distance_matrix.size
            
        # Safe matrix processing
        mat = np.copy(distance_matrix)
        np.fill_diagonal(mat, np.inf)
        mat[mat <= 0] = stability_factor
        
        # Sigmoid transformation
        with np.errstate(divide='ignore', invalid='ignore'):
            visibility = 1/(1 + np.exp(-0.5*(1/(mat + stability_factor))))
            
        # Dynamic scaling
        visibility = visibility / (visibility.max() + stability_factor)
        pheromone = np.ones_like(mat)
        
        # Final heuristics
        heuristic = (pheromone**alpha) * (visibility**beta)
        return heuristic/(heuristic.sum() + stability_factor)
        
    except Exception as e:
        print(f"Optimized heuristic error: {str(e)}")
        return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
    #EVOLVE-END