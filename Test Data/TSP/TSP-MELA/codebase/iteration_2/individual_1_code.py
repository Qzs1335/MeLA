import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix, beta=2.0):
    #EVOLVE-START
    def stable_softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)
    
    try:
        alpha = 1  # Standard pheromone coefficient
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        
        # Ensure proper distance matrix handling
        distance_matrix = np.where(distance_matrix > 0, distance_matrix, 1e-8)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Stable visibility calculation using safe log
        with np.errstate(divide='ignore', invalid='ignore'):
            log_dist = np.log(distance_matrix + 1e-16)
            visibility = np.exp(-log_dist)  # Inverse distance weights
        
        # If any invalid elements remain, make them minimally probable
        visibility = np.nan_to_num(visibility, nan=0.0, posinf=0.0, neginf=0.0)
        
        pheromone = np.ones_like(distance_matrix)
        heuristic = (pheromone**alpha) * (visibility**beta)
        
        # Add safe normalization with epsilon
        result = stable_softmax(heuristic * 10)
        if not np.all(np.isfinite(result)):
            raise ValueError("Result contains invalid values")
            
    except Exception as e:
        result = stable_softmax(np.ones_like(distance_matrix))
    
    return result
    #EVOLVE-END