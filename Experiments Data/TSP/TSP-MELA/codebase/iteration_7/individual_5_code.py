import numpy as np
import numpy as np
import warnings

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Handle null/empty inputs
        if distance_matrix is None or distance_matrix.size == 0:
            return np.ones_like(distance_matrix)/distance_matrix.shape[1] if distance_matrix else None
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Convert input to numeric values safely
            distance_matrix = np.nan_to_num(distance_matrix, nan=1.0, posinf=1.0, neginf=1.0)
            distance_matrix = np.array(distance_matrix, dtype=np.float64)
            
            # Set distance to self as infinite to prevent selection
            np.fill_diagonal(distance_matrix, np.inf)
            
            # Create visibility matrix numerically safely
            visible_distances = np.where(distance_matrix <= 0, np.nan, distance_matrix)
            vis_mean = np.nanmean(visible_distances, axis=1, keepdims=True)
            vis_min = np.nanmin(visible_distances, axis=1, keepdims=True)
            normalized_dist = np.nan_to_num(vis_min / visible_distances, nan=0.0)
            
            # Dynamic parameters based on distance statistics
            avg_dist = np.nanmean(visible_distances)
            scale_factor = 0.5 / max(avg_dist, 1e-5)
            alpha = 0.5 + (1.0 / (1 + np.exp(-scale_factor)))
            beta = 1.0 + scale_factor * 2
            
            # Safe softmax transformation
            probabilities = normalized_dist ** (beta) if avg_dist > 0 else np.ones_like(distance_matrix)
            probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
            probabilities = np.nan_to_num(probabilities, nan=1.0/distance_matrix.shape[1])
            
            return probabilities
            
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        sz = distance_matrix.shape if hasattr(distance_matrix, 'shape') else ()
        return np.ones(sz)/max(1, np.prod(sz)) if sz else None
    #EVOLVE-END