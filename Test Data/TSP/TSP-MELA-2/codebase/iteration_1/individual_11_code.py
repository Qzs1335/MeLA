import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    scaled = np.log(distance_matrix + epsilon)
    normalized = scaled / np.max(scaled)
    return (1 / (distance_matrix + epsilon)) * (0.5 + 0.5*normalized)
    #EVOLVE-END       
    return 1 / distance_matrix