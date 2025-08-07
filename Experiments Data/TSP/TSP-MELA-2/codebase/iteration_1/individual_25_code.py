import numpy as np
import numpy as np 
def heuristics_v2(data_al, data_pb, Positions, Best_pos, Best_score, rg):      
    #EVOLVE-START
    epsilon = 1e-8
    scaled_dist = np.log(distance_matrix + 1) + epsilon
    decay_factor = np.exp(-0.1 * distance_matrix)
    return (1/scaled_dist) * decay_factor        
    #EVOLVE-END       
    return Positions