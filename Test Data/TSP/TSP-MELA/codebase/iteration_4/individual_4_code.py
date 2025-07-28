import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1        
        beta = 2         
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            return None
            
        np.fill_diagonal(distance_matrix, np.inf)
        scale_factor = np.median(distance_matrix[distance_matrix != np.inf])
        epsilon = max(1e-10, scale_factor/1e6)
        
        visibility = np.log(1 + 1/(distance_matrix + epsilon))
        pheromone = np.exp(-distance_matrix/scale_factor)  
        
        heuristic = (pheromone**alpha) * (visibility**beta)
        normalized = heuristic/(np.sum(heuristic, axis=1, keepdims=True) + 1e-10)
        return normalized
        
    except Exception as e:
        print(f"Heuristic error: {e}")
        return np.ones_like(distance_matrix)/distance_matrix.size if distance_matrix.size>0 else None
    #EVOLVE-END