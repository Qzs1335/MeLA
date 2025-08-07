import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    epsilon = 1e-10  
    return (0.7 * np.exp(-distance_matrix)) + (0.3 / (distance_matrix + epsilon))  # Smoother decay & global awareness
    #EVOLVE-END       
    return 1 / distance_matrix