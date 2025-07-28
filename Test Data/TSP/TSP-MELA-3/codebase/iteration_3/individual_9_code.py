import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5  # More practical magnitude for TSP scales
    return 2.0 ** (1/(distance_matrix + epsilon))  # Smoother than exp via base adjustment
    #EVOLVE-END       
    return 1 / distance_matrix