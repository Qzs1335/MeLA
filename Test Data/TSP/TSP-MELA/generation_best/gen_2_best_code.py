import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1         
        beta = 2
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        np.fill_diagonal(distance_matrix, np.inf)
        visibility = 1 / np.log1p(distance_matrix)  # log smoothing
        heuristic = visibility**beta  # simplified weighting
        
        # Weight normalization
        normalized = heuristic / (np.sum(heuristic) + 1e-20)
        return normalized
        
    except Exception as e:
        return np.ones_like(distance_matrix) / distance_matrix.size
    #EVOLVE-END