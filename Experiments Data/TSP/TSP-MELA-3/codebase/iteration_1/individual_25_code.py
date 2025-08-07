import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    n = distance_matrix.shape[0]
    centrality = np.exp(-distance_matrix.sum(axis=1)/n)
    return (np.sqrt(1 / distance_matrix) + 0.5*centrality.reshape(-1,1)) / 1.5
    #EVOLVE-END       
    return 1 / distance_matrix