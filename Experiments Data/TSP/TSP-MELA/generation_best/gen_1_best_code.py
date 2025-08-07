import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1         # pheromone influence
        beta = 2          # distance influence
        
        # Convert to numpy array if not already and validate input
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix provided")
            
        np.fill_diagonal(distance_matrix, np.inf)  # avoid division by zero
        visibility = 1 / (distance_matrix + 1e-10)
        pheromone = np.ones_like(distance_matrix)  # initial pheromones
        
        # Calculate heuristic function
        heuristic = np.power(pheromone, alpha) * np.power(visibility, beta)
        
        # Safe normalization
        min_val = np.min(heuristic)
        max_val = np.max(heuristic)
        
        if min_val == max_val:
            # All values are identical - return uniform weights
            normalized = np.ones_like(heuristic) / heuristic.size
        else:
            # Standard min-max normalization
            normalized = (heuristic - min_val) / (max_val - min_val + 1e-10)
            
        return normalized
        
    except Exception as e:
        print(f"Error in heuristics_v2: {str(e)}")
        # Return uniform weights as fallback
        return np.ones_like(distance_matrix) / distance_matrix.size if distance_matrix.size > 0 else None
    #EVOLVE-END