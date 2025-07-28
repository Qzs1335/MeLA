import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    scaled = np.log(1 + 1/distance_matrix)
    return scaled / np.max(scaled)
    #EVOLVE-END       
    return 1 / distance_matrix