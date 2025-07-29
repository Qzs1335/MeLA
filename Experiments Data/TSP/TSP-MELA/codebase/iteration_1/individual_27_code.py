import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    min_dist = np.min(distance_matrix[distance_matrix > 0])
    norm_dist = distance_matrix/min_dist
    return (np.log(1 + 1/norm_dist))/(distance_matrix + 1e-10)
    #EVOLVE-END       
    return 1 / distance_matrix