import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1.2              # optimized pheromone weight
        beta = 1.8               # adjusted distance weight
        eps = np.finfo(float).eps # dynamic stability factor
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size < 2:
            raise ValueError("Invalid distance matrix")
            
        # Stabilized visibility matrix
        adjusted_dist = distance_matrix.copy()
        np.fill_diagonal(adjusted_dist, np.inf)
        adjusted_dist[adjusted_dist <= 0] = eps
        
        visibility = np.exp(-adjusted_dist)  # exponential transform
        pheromone = np.ones_like(adjusted_dist)
        
        # Temperature-scaled heuristic
        temp = np.median(adjusted_dist)
        heuristic = np.power(pheromone, alpha) * np.power(visibility, beta/temp)
        
        # Safe normalization
        norms = np.linalg.norm(heuristic, axis=1, keepdims=True)
        return np.where(norms>0, heuristic/norms, 1/len(heuristic))
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        size = len(distance_matrix) if hasattr(distance_matrix,'__len__') else 1
        return np.ones((size,size))/size if size>1 else np.array([[1.]])
    #EVOLVE-END