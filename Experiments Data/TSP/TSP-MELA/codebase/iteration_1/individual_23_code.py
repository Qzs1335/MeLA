import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    alpha = 0.1  # Randomization factor
    scale_factor = np.log1p(distance_matrix.mean()) 
    heur = (1 / (distance_matrix + 1e-10)) * scale_factor
    heur += alpha * np.random.rand(*distance_matrix.shape)
    #EVOLVE-END       
    return heur