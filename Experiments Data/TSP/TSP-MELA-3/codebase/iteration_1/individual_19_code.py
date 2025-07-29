import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    kernel_size = min(5, distance_matrix.shape[0]//2)
    smoothed = np.exp(-distance_matrix**2/(2*(0.2*np.max(distance_matrix))**2))
    rand_comp = 0.1 * np.random.rand(*distance_matrix.shape)
    return 1/(distance_matrix+1e-10) * smoothed + rand_comp
    #EVOLVE-END
    return 1 / distance_matrix