import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1         # pheromone influence 
        beta = 1.5        # balanced distance influence
        eps = 1e-8        # numerical stability factor
        
        distance_matrix = np.asarray(distance_matrix)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Optimized influence calculation
        visible = np.log(1 + 1/(distance_matrix + eps))
        pheromone = np.exp(-0.1 * distance_matrix)
        
        heuristic = np.power(pheromone, alpha) * np.power(visible, beta)
        return (heuristic - np.min(heuristic)) / (np.ptp(heuristic) + eps)
        
    except Exception as e:
        print(f"Heuristics error: {e}")
        size = distance_matrix.shape[0] if distance_matrix.size > 0 else 1
        return np.ones(size)/size
    #EVOLVE-END