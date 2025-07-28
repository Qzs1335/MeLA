import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    np.fill_diagonal(distance_matrix, np.inf)
            
    # Core heuristic with log scaling
    heuristic = np.log1p(1/(distance_matrix + 1e-15))**2  # alpha=1, beta=2 compounded
    
    # More robust normalization
    heuristic -= np.min(heuristic)
    heuristic /= np.sum(heuristic) + 1e-15
    return heuristic
    #EVOLVE-END