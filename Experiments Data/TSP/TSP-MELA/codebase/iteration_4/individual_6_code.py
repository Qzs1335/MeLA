import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = np.log(len(distance_matrix))  # adaptive pheromone
        beta = 2 + np.sqrt(alpha/10)         # dynamic distance influence
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix provided")
            
        np.fill_diagonal(distance_matrix, np.inf)
        distance_matrix = np.clip(1 / distance_matrix, 1e-8, 1e8)  # bounded safety
        
        pheromone = (np.ones_like(distance_matrix) * 
                    np.exp(1 - np.eye(distance_matrix.shape[0])))  # decay on diagonal
        
        heuristic = np.power(distance_matrix * pheromone, beta) * alpha
        return heuristic / np.sum(heuristic, axis=1)[:, np.newaxis]
        
    except Exception as e:
        print(f"Error in heuristics_v2: {str(e)}")
        return np.ones_like(distance_matrix)/len(distance_matrix) if distance_matrix.size else None
    #EVOLVE-END