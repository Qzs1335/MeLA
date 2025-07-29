import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Dynamic weights and adaptive stability
        matrix_mean = np.mean(distance_matrix)
        alpha = min(2, 0.5 + matrix_mean/10)        # Dynamic pheromone weight 
        beta = max(1, 3 - matrix_mean/10)           # Dynamic distance weight
        stability_factor = max(1e-16, matrix_mean/1e10)
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.ndim != 2:
            raise ValueError("Invalid distance matrix dimensions")
            
        # Enhanced distance handling with exponential decay  
        distance_matrix = np.exp(-distance_matrix) + stability_factor 
        np.fill_diagonal(distance_matrix, 0)
        
        # Calculate heuristic with validation
        heuristic = distance_matrix
        total = np.sum(heuristic)
        if total > stability_factor:
            return (heuristic/total) ** (alpha + beta)
        return np.ones_like(heuristic) / heuristic.size
        
    except Exception as e:
        print(f"Optimized Heuristic Error: {str(e)}")
        safe_shape = distance_matrix.shape if hasattr(distance_matrix, 'shape') else (10,10)
        return np.ones(safe_shape)/np.prod(safe_shape)
    #EVOLVE-END