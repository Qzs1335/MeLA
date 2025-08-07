import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    distance_matrix = np.maximum(distance_matrix, 1e-100)  # Ensure positive distances
    np.fill_diagonal(distance_matrix, np.inf)  # Prevent self-loops
    
    # Safe computation of inverse distance with clipping
    inv_dist = 1 / distance_matrix
    inv_dist = np.clip(inv_dist, 0, 1e100)  # Prevent division explosions
    
    # Compute balance factor with safe operations
    mean_dist = np.mean(distance_matrix[distance_matrix < np.inf])
    max_dist = np.max(distance_matrix[distance_matrix < np.inf])
    balance = mean_dist / max_dist if max_dist > 0 else 1.0
    
    # Compute heuristic weights with numerical safety
    heuristic = np.power(inv_dist, 1 + balance * np.sign(inv_dist))
    heuristic = np.where(np.isfinite(heuristic), heuristic, 0)  # Replace inf/nan
    
    # Normalize with safety checks
    weight_sum = np.sum(heuristic, axis=1, keepdims=True)
    heuristic = np.where(weight_sum > 0, heuristic/weight_sum, 1.0/heuristic.shape[1])
    
    # Final clipping to valid probability range with margin
    return np.clip(heuristic, 1e-10, 1-1e-10)  # Numerically stable probabilities
    #EVOLVE-END