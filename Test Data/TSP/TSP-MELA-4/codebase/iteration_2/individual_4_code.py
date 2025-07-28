import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5  # prevents division by zero
    beta = 1.5      # optimized heuristic importance
    return np.exp(-distance_matrix/5) * ((1/(distance_matrix + epsilon))**beta)
    #EVOLVE-END