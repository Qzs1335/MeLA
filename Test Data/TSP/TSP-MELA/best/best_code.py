import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1 + np.random.rand()       # Dynamic pheromone weight  
        beta = 2 + np.random.rand()        # Adaptive distance weight
        stability_factor = 1e-16    
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Robust visibility calculation  
        visibility = np.clip(1/distance_matrix, 0, 1e16)  
        
        # Stabilized transforms
        log_phero = np.log(1 + stability_factor + np.ones_like(visibility))
        log_vis = np.log(visibility + stability_factor) 
        heuristic = np.exp(alpha * log_phero + beta * log_vis)
        
        # Min-max normalized output  
        h_min, h_max = np.min(heuristic), np.max(heuristic)
        return (heuristic - h_min)/(h_max - h_min + 1e-16)
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        return np.ones(distance_matrix.shape)/distance_matrix.size if distance_matrix.size > 0 else None
    #EVOLVE-END