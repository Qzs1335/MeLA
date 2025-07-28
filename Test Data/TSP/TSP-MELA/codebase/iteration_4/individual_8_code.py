import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1
        beta = 2
        
        # Input processing and safeguards
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0 or np.all(distance_matrix <= 0):
            raise ValueError("Invalid distance matrix")
            
        np.fill_diagonal(distance_matrix, np.inf)
        adaptive_eps = 1e-10 * np.log(distance_matrix.size) if distance_matrix.size > 1 else 1e-10
            
        # Core heuristic computation
        visibility = 1 / (distance_matrix + adaptive_eps)
        pheromone = np.ones_like(distance_matrix)
        heuristic = np.exp(alpha*np.log(pheromone) + beta*np.log(visibility + 1e-100))
        softmax_heuristic = np.exp(heuristic - np.max(heuristic))
        return softmax_heuristic / softmax_heuristic.sum()
        
    except Exception as e:
        print(f"Optimized heuristics error: {str(e)}")
        fallback = np.ones_like(distance_matrix)
        return fallback / fallback.sum() if distance_matrix.size > 0 else None
    #EVOLVE-END