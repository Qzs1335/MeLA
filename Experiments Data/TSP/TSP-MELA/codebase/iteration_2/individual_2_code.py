import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        adaptive_alpha = max(1, int(np.log2(distance_matrix.shape[0])))  # Scale with problem size
        beta = 2
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix")
            
        np.fill_diagonal(distance_matrix, np.inf)
        visibility = np.log1p(1/(distance_matrix + 1e-10*(distance_matrix.size**0.5)))
        pheromone = np.ones_like(distance_matrix)
        
        heuristic = (pheromone**adaptive_alpha) * (visibility**beta)
        norm_heuristic = np.log1p(heuristic)  # Logarithmic normalization
        return norm_heuristic/np.sum(norm_heuristic)  # Probabilistic scaling
        
    except Exception:
        return np.ones_like(distance_matrix)/distance_matrix.size if distance_matrix.size>0 else None
    #EVOLVE-END