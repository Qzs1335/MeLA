import numpy as np
import numpy as np
from scipy import stats

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Dynamic parameter adjustment
        std_dev = np.std(distance_matrix[np.isfinite(distance_matrix)])
        alpha = 1  # Fixed pheromone weight
        beta = max(2 + np.log1p(std_dev), 1)  # Adaptive distance sensitivity
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        np.fill_diagonal(distance_matrix, np.nan)
        
        # Log-scaled visibility with clipped values
        visibility = np.log1p(1/(distance_matrix + 1e-12))
        pheromone = np.ones_like(distance_matrix)
        
        heuristic = pheromone**alpha * visibility**beta
        normalized = stats.zscore(heuristic, nan_policy='omit')
        normalized = np.nan_to_num(normalized, posinf=0, neginf=0)
        return np.exp(normalized)  # Convert back to positive scale
        
    except Exception as e:
        print(f"Heuristic computation failed: {e}")
        return np.ones_like(distance_matrix)/distance_matrix.size if hasattr(distance_matrix, 'size') else None
    #EVOLVE-END