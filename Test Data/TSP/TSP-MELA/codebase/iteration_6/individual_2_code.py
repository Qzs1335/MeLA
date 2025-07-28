import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Dynamic parameters
        alpha_min, alpha_max = 0.5, 2.0
        beta_min, beta_max = 1.5, 3.0
        alpha = alpha_min + (alpha_max-alpha_min)*np.random.rand()
        beta = beta_min + (beta_max-beta_min)*np.random.rand()
        
        # Adaptive stability
        epsilon = np.finfo(np.float64).eps * distance_matrix.size
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix")
            
        # Exponential handling of distances
        distance_matrix[distance_matrix <= epsilon] = epsilon
        visibility = np.exp(-distance_matrix)
        
        # Adaptive pheromone decay
        pheromone = np.exp(-distance_matrix/distance_matrix.mean())
        
        # Combined heuristic with log stability
        heuristic = visibility**beta * pheromone**alpha
        heuristic = np.nan_to_num(heuristic, nan=epsilon)
        
        # Softmax normalization
        return np.exp(heuristic) / np.sum(np.exp(heuristic))
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
    #EVOLVE-END