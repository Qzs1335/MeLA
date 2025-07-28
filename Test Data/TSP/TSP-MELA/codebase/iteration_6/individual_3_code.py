import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Dynamic weighting with exponential decay
        beta = 2 - np.exp(-0.5 * np.mean(distance_matrix)) 
        alpha = 1 + np.log1p(np.max(distance_matrix))
        
        # Adaptive stability factor
        eps = max(1e-16, 1e-10 * np.median(distance_matrix))
        dm = np.asarray(distance_matrix, dtype=np.float64)
        np.fill_diagonal(dm, np.inf)
        dm = np.maximum(dm, eps)
        
        # Enhanced visibility calculation
        visibility = np.exp(-0.1 * dm)
        pheromone = np.ones_like(dm)
        
        # Stable exponentiation
        heuristic = np.power(pheromone, alpha) * np.power(visibility, beta)
        return heuristic / (np.sum(heuristic) + eps)
        
    except Exception:
        shape = getattr(distance_matrix,'shape',(1,))
        return np.ones(shape)/np.prod(shape)
    #EVOLVE-END