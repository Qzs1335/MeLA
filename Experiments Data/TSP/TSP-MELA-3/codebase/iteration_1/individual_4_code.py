import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    rng = np.random.default_rng()
    noise = 1 + 0.1*rng.random(distance_matrix.shape)
    return noise * np.log(1 + 1/(distance_matrix + 1e-10))
    #EVOLVE-END
    return 1 / distance_matrix