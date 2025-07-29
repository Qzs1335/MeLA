import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha, beta = 1.0, 2.0  # Using floats for numeric stability
        safe_dist = np.asarray(distance_matrix, dtype=np.float64)
        
        # Handle empty/invalid inputs
        if safe_dist.size == 0 or np.all(np.isnan(safe_dist)):
            raise ValueError("Invalid distance matrix")
            
        # Handle zeros and small distances safely
        np.fill_diagonal(safe_dist, np.inf)
        non_zero_mask = (safe_dist > 0)
        visibility = np.zeros_like(safe_dist)
        visibility[non_zero_mask] = 1 / safe_dist[non_zero_mask]  # Direct inverse for stable values
        visibility += 1  # Add 1 before log avoids log(1+1/e) edge cases
        visibility = np.log(visibility)
        
        # Pheromone calculation with max distance clipping 
        max_dist = np.nanmax(safe_dist)
        pheromone = np.exp(-safe_dist/(max_dist + 1e-10))
        
        # Combined heuristic with zero-division protection
        heuristic = (pheromone**alpha * visibility**beta) 
        
        # Soft normalization with safeguards
        h_min, h_max = np.nanmin(heuristic), np.nanmax(heuristic)
        if h_min == h_max or np.isnan(h_min) or np.isnan(h_max):
            raise ValueError("Degenerate heuristic range")
            
        normalized = (heuristic - h_min)/(h_max - h_min + 1e-10)
        return np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        
    except Exception:
        # Fallback uniform distribution
        size = distance_matrix.size if hasattr(distance_matrix, 'size') else len(distance_matrix)**2
        return np.ones_like(distance_matrix)/max(1, size)
    #EVOLVE-END