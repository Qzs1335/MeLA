import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Dynamic parameters based on matrix statistics
        mean_dist = np.mean(distance_matrix)
        alpha = 1 + np.log1p(mean_dist/100)  # adaptive pheromone weight
        beta = 2 / (1 + np.exp(-mean_dist/50))  # logistic scaling
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        
        # Numerical safeguards
        dist_clipped = np.where(distance_matrix <= 0, 1e-10, distance_matrix) 
        sigma = mean_dist/10  # dynamic epsilon scaling
        visibility = np.exp(-dist_clipped/(2*sigma + 1e-10))
        pheromone = np.ones_like(distance_matrix)
        
        heuristic = np.power(pheromone,alpha) * np.power(visibility,beta)
        return (np.exp(heuristic) / np.sum(np.exp(heuristic), axis=1)[:,None])  # row-wise softmax
        
    except Exception:
        uniform = np.ones_like(distance_matrix)/distance_matrix.shape[0]
        return np.exp(uniform) / np.sum(np.exp(uniform))
    #EVOLVE-END