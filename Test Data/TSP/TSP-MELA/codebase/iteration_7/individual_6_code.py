import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        stability_factor = 1e-16
        distance_matrix = np.array(distance_matrix, dtype=np.float64)
        
        # Stable visibility
        dist = distance_matrix.copy()
        dist[dist == 0] = stability_factor
        visibility = 1/(dist + stability_factor)
        visibility = np.nan_to_num(visibility, copy=False)
        
        # Balanced heuristic
        pheromone = np.ones_like(distance_matrix)
        heuristic = visibility ** 0.5 * pheromone ** 0.5
        
        # Normalized
        total = np.sum(heuristic)
        if abs(total) < stability_factor*1e3:
            return np.ones(distance_matrix.shape)/distance_matrix.size 
        return heuristic/total        
    except:
        return np.ones(distance_matrix.shape)/distance_matrix.size
    #EVOLVE-END