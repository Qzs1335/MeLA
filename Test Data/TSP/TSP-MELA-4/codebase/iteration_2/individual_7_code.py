import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5
    alpha = 1.5
    beta = 2.0
    return alpha * np.exp(-distance_matrix/5) * ((1/(distance_matrix + epsilon))**beta) * np.log1p(distance_matrix)
    #EVOLVE-END
    return 1 / distance_matrix