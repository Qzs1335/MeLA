import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    pheromone_weight = 2.0
    decay_factor = 0.1
    random_comp = 0.3 * np.random.rand(*distance_matrix.shape)
    return (1 / distance_matrix)**pheromone_weight * np.exp(-decay_factor * distance_matrix) + random_comp
    #EVOLVE-END       
    return 1 / distance_matrix