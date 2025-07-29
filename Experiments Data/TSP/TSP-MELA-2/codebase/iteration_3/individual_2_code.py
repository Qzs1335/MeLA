import numpy as np
import numpy as np 
def heuristics_v2(data_al, data_pb, Positions, Best_pos, Best_score, rg):
    distance_matrix = np.linalg.norm(Positions[:, None, :] - Positions, axis=2)
    #EVOLVE-START
    eps = 1e-8
    log_scaled = np.log1p(distance_matrix.clip(eps))
    neigh_dist = np.partition(distance_matrix, 1)[:, 1:5]
    neigh_weights = np.median(neigh_dist, axis=1)[:,None]
    exp_terms = np.exp(-distance_matrix/neigh_weights)
    return (exp_terms / (distance_matrix + eps)) * neigh_weights / log_scaled
    #EVOLVE-END       
    return 1 / distance_matrix