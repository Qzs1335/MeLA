import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    avg_dist = np.mean(distance_matrix)
    node_degree = 1/(np.sum(distance_matrix, axis=1)+1e-10)
    exploration = np.random.rand(*distance_matrix.shape)*0.1
    return (1/distance_matrix) * node_degree[:,None] + exploration
    #EVOLVE-END       
    return 1 / distance_matrix