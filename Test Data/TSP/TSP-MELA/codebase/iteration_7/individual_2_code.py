import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Adaptive parameters
        n = distance_matrix.shape[0] if hasattr(distance_matrix, 'shape') else 1
        alpha = 1 + np.log(n)/5  # dynamic pheromone weight
        beta = 2 - np.log(n)/10  # dynamic distance weight
        stability_factor = 1e-16
        
        # Early return for small matrices
        if n <= 1:
            return np.ones(1) if n == 1 else None
            
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 1:
            return np.ones(1)
        
        # Handle numerical edge cases
        distance_matrix = np.where(distance_matrix <= stability_factor, stability_factor, distance_matrix)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Vectorized operations
        visibility = np.reciprocal(distance_matrix, where=distance_matrix > 0)
        visibility = np.nan_to_num(visibility, nan=0.0, posinf=0.0, neginf=0.0)
        
        pheromone = np.ones_like(distance_matrix)
        log_sum = alpha * np.log(pheromone + stability_factor) + beta * np.log(visibility + stability_factor)
        
        heuristic = np.exp(log_sum - np.max(log_sum))  # Numerical stability
        total = np.sum(heuristic)
        
        return heuristic/total if total > stability_factor else np.ones_like(heuristic)/heuristic.size
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
    #EVOLVE-END