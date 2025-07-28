import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10  # improved precision
    alpha = 1.5      # stronger pheromone weight
    beta = 1.5       # balanced heuristic weight
    T = 8            # temperature control
    return alpha * np.exp(-distance_matrix/T) * ((1/(distance_matrix + epsilon))**beta)
    #EVOLVE-END
    return 1 / distance_matrix