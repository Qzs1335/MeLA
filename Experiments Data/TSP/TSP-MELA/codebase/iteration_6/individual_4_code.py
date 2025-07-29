import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Dynamic weights based on matrix properties
        beta = 2 - np.clip(np.log10(np.median(distance_matrix[distance_matrix > 0])), 0.5, 1.5)
        alpha = 2/beta  # Compensatory weight
        
        # Adaptive numerical stability
        epsilon = np.finfo(float).eps * np.max(distance_matrix)
        
        # Process matrix with safety
        dist = np.where((distance_matrix > 0) | (np.eye(len(distance_matrix), dtype=bool)), 
                       distance_matrix + epsilon, 
                       epsilon)
        np.fill_diagonal(dist, np.inf)
        
        # Core heuristic calculation
        vis = np.exp(-beta * np.log(dist))
        phero = np.exp(alpha * np.log(np.ones_like(dist) + epsilon))
        heuristic = np.multiply(phero, vis)
        
        # Safe normalization
        softmax = heuristic / (np.sum(heuristic) + epsilon)
        return np.nan_to_num(softmax, nan=1/len(softmax))
    
    except Exception:
        size = len(distance_matrix) if hasattr(distance_matrix, '__len__') else 10
        return np.ones((size, size)) / (size**2)
    #EVOLVE-END