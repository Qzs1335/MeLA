import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1         # maintain optimal pheromone weight
        beta = 2          # preserve ideal distance influence
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix")
            
        np.fill_diagonal(distance_matrix, np.inf)
        visibility = 1 / np.log1p(distance_matrix)  # logarithmic scaling
        
        heuristic = visibility ** beta  # simplified since pheromone=1 initially
        
        # Exponential transformation for better non-linearity
        exp_heuristic = np.exp(heuristic - np.max(heuristic))
        if not np.any(exp_heuristic):  # prevent all zeros
            exp_heuristic = np.ones_like(heuristic)
            
        return exp_heuristic / exp_heuristic.sum()  # probabilistic normalization
        
    except Exception:
        # Symmetric fallback matrix preserves structure
        size = len(distance_matrix) if hasattr(distance_matrix, '__len__') else 1
        return np.ones((size, size)) / size
    #EVOLVE-END