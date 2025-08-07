import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Adaptive weights based on matrix properties
        mean_dist = np.nanmean(distance_matrix[distance_matrix > 0])
        beta = max(1, min(3, 2 * (1 + np.log1p(mean_dist))))  # Dynamic distance weight [1,3]
        alpha = 4 - beta                                       # Complementary pheromone weight
        
        stability_factor = 1e-16
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        
        # Process distance matrix
        valid_mask = ~np.eye(distance_matrix.shape[0], dtype=bool)
        distance_matrix[~valid_mask] = np.inf
        distance_matrix[distance_matrix <= 0] = stability_factor
        
        # Enhanced visibility with sigmoid
        visibility = 1 / (distance_matrix + stability_factor)
        visibility = np.clip(visibility, 0, 1e16)
        
        # Heuristic composition
        log_terms = alpha * np.log1p(visibility) + beta * np.log1p(visibility**2)
        heuristic = np.exp(log_terms - np.max(log_terms))  # Stable exponentiation
        
        # Sigmoid normalization
        heuristic = 1 / (1 + np.exp(-heuristic))
        return heuristic / np.sum(heuristic)
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
    #EVOLVE-END