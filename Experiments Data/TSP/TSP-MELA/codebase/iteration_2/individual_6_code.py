import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Constants with safer ranges
        alpha = 1.0        # Pheromone influence
        base_beta = 1.5    # Base heuristic influence
        MIN_DIST = 1e-10   # Minimum distance to prevent numerical issues
        
        # Validate and process input
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0 or not np.all(np.isfinite(distance_matrix)):
            raise ValueError("Invalid distance matrix")
            
        # Create clean distance matrix
        clean_dist = distance_matrix.copy()
        np.fill_diagonal(clean_dist, np.inf)  # Set diagonal to infinity
        clean_dist = np.maximum(clean_dist, MIN_DIST)  # Ensure no negative or zero values
        
        # Safe normalization and visibility calculation
        max_dist = np.max(clean_dist)
        normalized_dist = np.clip(clean_dist / max_dist, MIN_DIST, None)
        
        # Logarithm with clipped input and scaled output
        visibility = 1 - np.log(normalized_dist)  # Invert for better scaling
        
        # Adaptive beta with bounds
        size_factor = np.log10(distance_matrix.size) / 20
        beta = base_beta * (1 + np.clip(size_factor, -0.5, 0.5))
        
        # Final heuristic with numerical stability
        with np.errstate(over='ignore', under='ignore'):
            heuristic = np.exp(-alpha * normalized_dist) * (visibility**beta)
            heuristic = np.nan_to_num(heuristic, nan=1.0, posinf=1.0, neginf=1.0)
            
        # Standardization with safeguards
        mean_h = np.mean(heuristic)
        std_h = np.std(heuristic)
        return (heuristic - mean_h) / (std_h + 1e-15)
        
    except Exception as e:
        print(f"Stable heuristics error: {e}")
        return np.ones_like(distance_matrix) if isinstance(distance_matrix, np.ndarray) else None
    #EVOLVE-END