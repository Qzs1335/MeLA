import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    scaled_dist = np.where(distance_matrix < 10, 
                          np.exp(-0.2 * distance_matrix),
                          np.log(1/(distance_matrix + epsilon)))
    #EVOLVE-END
    return scaled_dist