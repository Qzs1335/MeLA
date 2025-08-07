import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1         # pheromone influence
        beta = 2          # distance influence
        
        # Input validation
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix provided")
            
        np.fill_diagonal(distance_matrix, np.inf)  # avoid division by zero
        visibility = np.log(1 + 1 / (distance_matrix + 1e-10))  # logarithmic scaling
        pheromone = np.ones_like(distance_matrix)
        
        # Heuristic calculation with softmax normalization
        heuristic = np.power(pheromone, alpha) * np.power(visibility, beta)
        normalized = np.exp(heuristic)/np.sum(np.exp(heuristic))
        
        return normalized
        
    except Exception as e:
        print(f"Error in heuristics_v2: {str(e)}")
        return np.ones_like(distance_matrix)/distance_matrix.size if distance_matrix.size > 0 else None
    #EVOLVE-END