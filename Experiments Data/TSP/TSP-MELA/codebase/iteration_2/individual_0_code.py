import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1  
        beta = 2  
        eps = 1e-10 * np.prod(distance_matrix.shape)  # Adaptive epsilon
        
        dm = np.asarray(distance_matrix, dtype=np.float64)
        np.fill_diagonal(dm, np.inf)
        
        # Logarithmically scaled visibility
        visibility = 1 / (np.log1p(dm) + eps)  
        pheromone = np.ones_like(dm)
        heuristic = pheromone**alpha * visibility**beta
        
        # Softmax normalization
        exp_vals = np.exp(heuristic - np.max(heuristic))
        normalized = exp_vals / (np.sum(exp_vals) + eps)
        
        return normalized

    except (ValueError, ArithmeticError) as e:
        print(f"Heuristic error: {str(e)}")
        size = distance_matrix.size if hasattr(distance_matrix, 'size') else 1
        return np.ones(size)/size if size > 0 else None
    #EVOLVE-END