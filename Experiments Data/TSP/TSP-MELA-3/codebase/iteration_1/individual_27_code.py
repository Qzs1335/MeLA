import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    alpha = 1.0  # pheromone weight
    beta = 2.0   # heuristic weight 
    temp = 0.1   # exploration temperature
    
    H = np.zeros_like(distance_matrix)
    pheromone = np.ones_like(distance_matrix)
    np.fill_diagonal(pheromone, 0)
    
    H = (pheromone**alpha)  * ((1/(distance_matrix + 1e-8))**beta)
    mask = np.random.rand(*H.shape) < temp
    H[mask] = np.random.rand(np.sum(mask))
    #EVOLVE-END       
    return H