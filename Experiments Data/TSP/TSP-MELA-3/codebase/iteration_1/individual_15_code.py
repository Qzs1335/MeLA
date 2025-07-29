import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    inverse_dist = 1 / (distance_matrix + 1e-10)
    random_component = np.random.uniform(0, 0.3, size=distance_matrix.shape)
    combined = 0.7 * (inverse_dist/inverse_dist.max()) + 0.3 * random_component
    return combined
    #EVOLVE-END       
    return 1 / distance_matrix