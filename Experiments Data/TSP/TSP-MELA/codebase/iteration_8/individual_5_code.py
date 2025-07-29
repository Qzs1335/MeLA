import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Dynamic parameters based on matrix properties
        matrix_mean = np.mean(distance_matrix[distance_matrix > 0])
        alpha = 1 + np.log1p(matrix_mean) 
        beta = 2 - np.log1p(matrix_mean)
        stability_factor = 1e-16
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix")
            
        distance_matrix[distance_matrix == 0] = stability_factor
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Sigmoid scaling for stability
        visibility = 1 / (1 + np.exp(-1/(distance_matrix + stability_factor)))
        
        # Adaptive minimum checking
        min_val = np.min(visibility[visibility > 0])
        pheromone = np.ones_like(distance_matrix) * min_val
        
        # Scaled combination
        heuristic = (pheromone**alpha) * (visibility**beta)
        
        # Diversity-based normalization
        if np.ptp(heuristic) > stability_factor:
            return heuristic / np.sum(heuristic)
        return np.ones_like(heuristic) / heuristic.size
        
    except Exception as e:
        print(f"Adaptive Heuristic Error: {str(e)}")
        size = distance_matrix.shape[0] if hasattr(distance_matrix, 'shape') else 10
        return np.ones((size,size))/size**2
    #EVOLVE-END