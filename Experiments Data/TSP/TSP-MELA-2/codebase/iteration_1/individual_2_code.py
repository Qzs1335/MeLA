import numpy as np
import numpy as np
from scipy.sparse.csgraph import shortest_path

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    n = len(distance_matrix)
    # Add small constant to avoid division by zero
    safe_distance = distance_matrix + 1e-10
    
    centrality = np.zeros(n)
    for i in range(n):
        paths = shortest_path(distance_matrix, directed=False, indices=i, return_predecessors=False)
        mask = np.isfinite(paths)
        # Calculate reachable nodes excluding self
        reachable = mask.sum(axis=0) - 1 # subtract self
        valid_paths = mask.sum() - n # total paths excluding self
        if valid_paths > 0:
            centrality += reachable / valid_paths
        else:
            centrality += 0
    
    distance_comp = (1 / safe_distance) * 0.7
    centrality_comp = 0.3 * (centrality.reshape(-1,1) + centrality)
    
    # Ensure no NaN in final output
    result = distance_comp + centrality_comp
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    return result
    #EVOLVE-END