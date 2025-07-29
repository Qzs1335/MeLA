import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    nn_influence = np.min(distance_matrix, axis=1, keepdims=True)
    beta = 2.5  # Exploration parameter
    return (1 / (distance_matrix + 1e-10)) * np.exp(-beta * (distance_matrix/nn_influence))
    #EVOLVE-END
    return 1 / distance_matrix