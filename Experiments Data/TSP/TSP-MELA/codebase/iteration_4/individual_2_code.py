import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1       # pheromone factor(keep from optimized)
        beta = 2        # distance factor(keep from optimized)
        epsilon = 1e-10 # safeguard
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        np.clip(distance_matrix, epsilon, None, out=distance_matrix)
        
        visibility = 1 / distance_matrix
        visibility = np.log(1 + visibility)  # enhanced contrast
        pheromone = np.ones_like(distance_matrix)
        
        heuristic = np.power(pheromone, alpha) * np.power(visibility, beta)
        np.maximum(heuristic, 0, out=heuristic)  # enforce non-negativity
        
        return np.exp(heuristic) / np.sum(np.exp(heuristic))  # softmax
        
    except Exception as e:
        print(f"Optimized error: {str(e)}")
        size = len(distance_matrix) if hasattr(distance_matrix, '__len__') else 1
        return np.full(size, 1/size) if size > 0 else None
    #EVOLVE-END