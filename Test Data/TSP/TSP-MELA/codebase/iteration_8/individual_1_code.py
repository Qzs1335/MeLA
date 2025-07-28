import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Adaptive parameters based on matrix properties
        matrix_mean = np.mean(distance_matrix)
        alpha = max(1, 0.5 * np.log1p(matrix_mean))
        beta = min(2, 1 + np.sqrt(matrix_mean))
        stability_factor = max(1e-16, 1e-8 * matrix_mean)
        
        # Input validation & preprocess
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0 or distance_matrix.ndim != 2:
            raise ValueError("Invalid distance matrix")
        
        # Safe matrix transformations
        mask = np.eye(distance_matrix.shape[0], dtype=bool)
        adj_matrix = np.where(mask, np.inf, 
                             np.clip(distance_matrix, stability_factor, 1e16))
        
        visibility = np.exp(-np.log(adj_matrix)/beta)  # Stable inverse
        
        pheromone = np.cos(np.pi*np.random.random(adj_matrix.shape)*0.1) + 1.1
        log_term = (alpha*np.log(pheromone)) + (beta*np.log(visibility))
        
        heuristic = np.exp(np.clip(log_term, -100, 100))
        norm_heuristic = np.nan_to_num(heuristic/heuristic.max(), nan=0)
        return norm_heuristic/norm_heuristic.sum()
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        return np.ones(distance_matrix.shape)/distance_matrix.size
    #EVOLVE-END       
    return distance_matrix