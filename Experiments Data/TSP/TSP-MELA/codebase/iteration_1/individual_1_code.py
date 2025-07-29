import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    return 1 / (np.log(distance_matrix + 1e-10) + 1)
    #EVOLVE-END
    return 1 / distance_matrix