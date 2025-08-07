import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10  # Prevent division by zero
    scaled = np.log(1 + distance_matrix)
    heuristic = np.exp(-0.5 * scaled)
    heuristic = heuristic / (heuristic.max() + epsilon)
    #EVOLVE-END       
    return heuristic