import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        beta = 5  # Increased distance weight preference
        epsilon = 1e-12  # Micro stability factor
        
        d = np.asarray(distance_matrix, dtype=np.float64)        
        np.fill_diagonal(d, np.inf)  # Block self-transitions
        
        # Optimized visibility calculation
        vis = np.divide(1, np.maximum(d, epsilon), 
                       where=((d!=0) & (~np.isinf(d))))
        
        # Single-pass normalized heuristic
        heuristic = np.where(vis>0, vis**beta, 0)
        divisor = np.sum(heuristic)
        
        return heuristic/divisor if divisor>0 else np.ones_like(heuristic)/heuristic.size
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        return np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None
    #EVOLVE-END