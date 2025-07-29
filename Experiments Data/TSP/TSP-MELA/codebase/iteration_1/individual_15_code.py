import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10  # Prevent division by zero
    power = 1.5      # Makes nearer neighbors more attractive
    return 1 / (distance_matrix + epsilon)**power
    #EVOLVE-END
    return 1 / distance_matrix