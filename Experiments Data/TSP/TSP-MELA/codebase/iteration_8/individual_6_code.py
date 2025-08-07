import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        # Dynamic parameters
        mat_mean = np.mean(distance_matrix[distance_matrix>0])
        stability_factor = max(1e-16, mat_mean*1e-8)
        sig_param = lambda x: 1/(1+np.exp(-0.1*x))
        
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        np.fill_diagonal(distance_matrix, np.inf)
        distance_matrix = np.maximum(distance_matrix, stability_factor)
        
        # Adaptive calculation
        vis_rank = np.argsort(np.argsort(distance_matrix))
        visibility = sig_param(1/distance_matrix)*vis_rank
        
        # Rank-based normalization
        ranks = np.argsort(visibility.ravel()) + 1
        inv_ranks = 1/ranks.reshape(visibility.shape)
        inv_ranks = inv_ranks/np.sum(inv_ranks)
        
        return inv_ranks
        
    except Exception as e:
        print(f"Heuristic Error: {str(e)}")
        return np.ones(distance_matrix.shape)/distance_matrix.size
    #EVOLVE-END