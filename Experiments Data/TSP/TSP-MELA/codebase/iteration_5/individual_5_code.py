import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 0.8              # adaptive pheromone weight
        beta = 2.2               # enhanced distance weight
        stability_factor = 1e-12 # improved numerical stability
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix")
            
        # Enhanced zero/inf handling with exponential decay
        distance_matrix = np.where(distance_matrix <= 0, np.exp(-6)*np.max(distance_matrix), distance_matrix)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Optimized visibility calculation
        visibility = np.exp(-distance_matrix/3) + stability_factor
        
        # Dynamic heuristic calculation
        pheromone = np.ones_like(distance_matrix)
        heuristic = (pheromone**alpha) * (visibility**beta)
        
        # Adaptive normalization
        if np.any(heuristic > 0):
            return heuristic/np.sum(heuristic)
        return np.ones_like(heuristic)/len(heuristic)
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        return np.ones(distance_matrix.shape[0])/distance_matrix.shape[0] if hasattr(distance_matrix, 'shape') else None
    #EVOLVE-END