import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Dynamic parameters
        progress = 0.5  # Simulated iteration progress (0-1)
        alpha = max(1, 3*(1-progress))  # Decaying pheromone influence
        beta = min(5, 2*(1+progress))   # Growing distance influence
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Adaptive epsilon prevents overflows
        epsilon = 1e-5 * (1 + progress)
        visibility = 1 / (distance_matrix + epsilon)
        
        # Pheromone with evaporation rate
        evaporation = 0.1
        pheromone = (1-evaporation)*np.ones_like(distance_matrix)  
        
        heuristic = np.exp(np.log(pheromone+epsilon)*alpha + np.log(visibility+epsilon)*beta)
        normalized = np.log1p(heuristic) / (np.sum(np.log1p(heuristic)) + epsilon)
        
        return normalized
        
    except Exception as e:
        print(f"Optimized heuristics error: {str(e)}")
        size = distance_matrix.shape[0] if isinstance(distance_matrix, np.ndarray) else len(distance_matrix)
        return np.ones(size) / size
    #EVOLVE-END