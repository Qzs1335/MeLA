import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        
        # Add small constant to avoid zeros and negatives
        adjusted_dist = distance_matrix + 1e-16
        np.fill_diagonal(adjusted_dist, np.inf)  # Always ignore self-distances
        
        # Safe reciprocal with clipping to prevent overflow
        safe_reciprocal = np.clip(1.0 / adjusted_dist, 1e-16, 1e16)
        
        # Calculate parameters with safeguards
        alpha = 1.0 + np.log1p(np.clip(np.mean(distance_matrix), 1e-16, None))
        beta = 2.0 / max(1e-8, 1 + np.std(distance_matrix))
        
        # Calculate weights stably
        log_weights = beta * np.log(safe_reciprocal)  # Log scale importance
        weights = np.exp(np.clip(log_weights, -700, 700))  # Anti-log with clipping
        
        # Normalize probabilities safely
        sum_weights = max(np.sum(weights), 1e-16)
        probabilities = weights / sum_weights
        
        # Final checks to handle any potential numerical issues
        probabilities = np.nan_to_num(probabilities, nan=1.0/weights.size)
        probabilities = np.clip(probabilities, 1e-8, 1)
        probabilities /= probabilities.sum()  # Renormalize
        
        return probabilities
    except Exception as e:
        print(f"Optimized heuristic failed: {e}")
        size = distance_matrix.shape[0] if hasattr(distance_matrix, 'shape') else len(distance_matrix)
        return np.ones(size)/size  # Return uniform distribution as fallback
    #EVOLVE-END