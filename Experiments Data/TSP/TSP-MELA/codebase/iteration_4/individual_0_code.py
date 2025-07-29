import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        alpha = 1         
        beta = 2          
        eps = 1e-5
        explore_prob = 0.1
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        if distance_matrix.size == 0:
            raise ValueError("Empty distance matrix")
            
        np.fill_diagonal(distance_matrix, np.max(distance_matrix)*1.1)
        
        # Logarithmic stabilization with adaptive scaling
        log_dist = np.log(distance_matrix + eps)
        visibility = 1/(log_dist - np.min(log_dist) + 1)
        
        # Softmax normalization
        heuristic = (visibility**beta) * (np.ones_like(distance_matrix)**alpha)
        heuristic_exp = np.exp(heuristic - np.max(heuristic))
        softmax = heuristic_exp / np.sum(heuristic_exp)
        
        # Epsilon-greedy exploration
        if np.random.rand() < explore_prob:
            return np.ones_like(distance_matrix)/distance_matrix.size
        return softmax
        
    except Exception as e:
        return np.ones_like(distance_matrix)/distance_matrix.size if distance_matrix.size > 0 else None
    #EVOLVE-END