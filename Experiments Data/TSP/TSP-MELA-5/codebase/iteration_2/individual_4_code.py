import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-6
    transformed = np.tanh(np.sqrt(distance_matrix)) * np.log1p(1/(distance_matrix + epsilon))
    return transformed / (np.median(transformed) + epsilon)
    #EVOLVE-END