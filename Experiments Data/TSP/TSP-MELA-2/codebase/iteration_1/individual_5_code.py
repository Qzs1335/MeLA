import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    node_degree = (distance_matrix > 0).sum(axis=0)
    centrality = np.outer(node_degree, node_degree)
    normalized = centrality / centrality.max()
    return (0.7 * normalized + 0.3 * (1 / distance_matrix))
    #EVOLVE-END        
    return 1 / distance_matrix