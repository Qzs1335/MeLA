import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10  # Avoid division by zero
    return 1 / np.log1p(distance_matrix + epsilon)  # Logarithmic scaling
    #EVOLVE-END
    return 1 / distance_matrix