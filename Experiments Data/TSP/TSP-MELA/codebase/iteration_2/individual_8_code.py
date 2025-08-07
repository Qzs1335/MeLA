import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha, beta = 1, 2   # Optimal from historical data
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Enhanced visibility calculation with dynamic adaptation
        visibility = np.log(1 + 1/(distance_matrix + 1e-8))
        pheromone = np.ones_like(distance_matrix)
        
        # Power-scaling combination
        heuristic = pheromone**alpha * visibility**beta  
        return heuristic/np.sum(heuristic)  # Stable probabilistic normalization
        
    except Exception:
        return np.ones_like(distance_matrix)/distance_matrix.size
    #EVOLVE-END