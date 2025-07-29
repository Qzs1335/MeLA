import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1.5              # optimized pheromone weight 
        beta = 2.5               # optimized distance weight
        eps = 1e-16              # numerical stability
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        np.fill_diagonal(distance_matrix, np.inf)
        visibility = np.exp(beta * np.log(1 / (distance_matrix + eps) + eps))
        
        heuristic = visibility * np.exp(alpha * np.log(0.95*np.ones_like(distance_matrix)+eps))
        total = np.sum(heuristic)
        return heuristic/total if total>0 else np.ones_like(heuristic)/heuristic.size
        
    except Exception:
        return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
    #EVOLVE-END