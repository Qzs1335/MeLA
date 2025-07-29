import numpy as np
import numpy as np  
def heuristics_v2(distance_matrix):     
    eps = 1e-8  
    return np.exp(-np.sqrt(distance_matrix)) / (distance_matrix + eps)