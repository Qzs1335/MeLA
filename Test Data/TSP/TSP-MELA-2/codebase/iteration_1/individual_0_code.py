import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    visibility = 1/(distance_matrix + epsilon)
    noise = 0.01 * np.random.rand(*distance_matrix.shape)
    return np.exp(-distance_matrix/10) * visibility + noise
    #EVOLVE-END
    return 1 / distance_matrix