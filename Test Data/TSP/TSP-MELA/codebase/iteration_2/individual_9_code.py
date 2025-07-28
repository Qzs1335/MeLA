import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Adaptive parameters based on input scale
        scale_factor = np.log10(np.median(distance_matrix) + 1)
        alpha = max(1, 1.5 - 0.3*scale_factor)  # pheromone decay 
        beta = min(3, 2.0 + 0.5*scale_factor)   # distance emphasis

        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Logarithmic scaling for better dynamic range
        visibility = np.log(1 + 1/(distance_matrix + 1e-10))
        pheromone = np.exp(-0.1 * distance_matrix)
        
        heuristic = np.clip(np.power(pheromone, alpha), 1e-10, 1e10) * np.power(visibility, beta)
        
        # Robust normalization
        q75 = np.percentile(heuristic, 75)
        normalized = np.clip(heuristic/q75, 0, 1) 
        
        return normalized
        
    except Exception:
        return np.ones_like(distance_matrix)/distance_matrix.size if distance_matrix.size > 0 else None
    #EVOLVE-END