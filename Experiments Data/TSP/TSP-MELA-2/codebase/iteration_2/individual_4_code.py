import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-5 * np.median(distance_matrix)
    k = max(3, int(0.2*distance_matrix.shape[0]))
    nearest = np.partition(distance_matrix, k, axis=1)[:,:k].mean(axis=1)
    sigmoid = 1 / (1 + np.exp(-distance_matrix/np.mean(distance_matrix)))
    return (sigmoid/(distance_matrix+eps)) * (0.7 + 0.3*(nearest[:,None]+nearest)/distance_matrix.mean())
    #EVOLVE-END
    return 1 / distance_matrix