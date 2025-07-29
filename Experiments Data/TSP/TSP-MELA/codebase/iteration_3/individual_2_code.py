import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1  # pheromone influence
        beta_adaptive = 2 - np.log1p(np.mean(distance_matrix))  # adaptive exploration
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Logarithmic scaling + adaptive offset
        visibility = 1 / np.log1p(distance_matrix + 1e-5)
        pheromone = np.ones_like(distance_matrix)
        heuristic = np.power(pheromone, alpha) * np.power(visibility, beta_adaptive)
        
        # Softmax normalization
        exp_vals = np.exp(heuristic - np.max(heuristic))
        return exp_vals / exp_vals.sum()
        
    except Exception as e:
        print(f"Optimized heuristic error: {e}")
        return np.ones_like(distance_matrix)/distance_matrix.size if hasattr(distance_matrix, 'size') else None
    #EVOLVE-END