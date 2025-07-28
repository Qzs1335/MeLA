import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 0.8              # optimized pheromone weight  
        beta = 2.5               # optimized distance weight
        stability_factor = 1e-16 # numerical stability
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix")
            
        # Handle zeros and infs
        mask = distance_matrix <= 0
        distance_matrix = np.where(mask, stability_factor, distance_matrix)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Efficient heuristic calculation
        heuristic = np.exp(alpha - beta * np.log(distance_matrix))
        heuristic = np.nan_to_num(heuristic, nan=0.0)
        
        # Improved normalization
        sum_h = np.sum(heuristic)
        return heuristic / sum_h if sum_h > 0 else np.ones_like(heuristic)/heuristic.size
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
    #EVOLVE-END