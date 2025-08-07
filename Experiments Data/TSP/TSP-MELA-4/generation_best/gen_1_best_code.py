import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5  # prevents division by zero
    alpha = 1.0     # pheromone importance
    beta = 2.0      # heuristic importance
    return np.exp(-distance_matrix/10) * ((1/(distance_matrix + epsilon))**beta)
    #EVOLVE-END
    return 1 / distance_matrix