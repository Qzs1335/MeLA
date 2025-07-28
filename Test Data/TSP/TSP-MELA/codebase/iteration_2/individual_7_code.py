import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1                        # pheromone weight (fixed)
        beta = 2                         # optimal from history
        
        distance_matrix = np.asarray(distance_matrix, np.float64)
        if distance_matrix.size == 0 or not np.allclose(distance_matrix, distance_matrix.T):
            raise ValueError("Invalid distance matrix")
            
        np.fill_diagonal(distance_matrix, np.inf)
        visibility = 1 / np.log(distance_matrix + np.e)  # logarithmic scaling
        
        # Dynamic calculation with stability checks
        heuristic = np.power(visibility, beta)
        mean = np.mean(heuristic)
        std = np.std(heuristic) + 1e-10
        
        return (heuristic - mean) / std  # z-score normalization
        
    except Exception as e:
        print(f"Heuristic error: {str(e)[:30]}...")
        size = distance_matrix.shape[0]
        return np.ones((size,size))/size if size>0 else None
    #EVOLVE-END