import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    # MST-based heuristic
    mst = np.zeros_like(distance_matrix)
    n = len(distance_matrix)
    selected = [0]
    candidate = set(range(1,n))
    
    while candidate:
        min_dist = np.inf
        u, v = -1, -1
        for node in selected:
            for other in candidate:
                if distance_matrix[node][other] < min_dist:
                    min_dist = distance_matrix[node][other]
                    u, v = node, other
        selected.append(v)
        candidate.remove(v)
        mst[u][v] = mst[v][u] = min_dist
        
    centrality = 1 / np.sum(distance_matrix, axis=1)
    return 0.7*(1 / distance_matrix) + 0.2*centrality[:,None] + 0.1*(1/(mst+1e-10))
    #EVOLVE-END       
    return 1 / distance_matrix