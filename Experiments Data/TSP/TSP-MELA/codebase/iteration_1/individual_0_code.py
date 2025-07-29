import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    n = distance_matrix.shape[0]
    normalized_dist = distance_matrix / np.max(distance_matrix)
    
    # Initialize bonus matrix with zeros
    nn_bonus = np.zeros_like(distance_matrix)
    # Get indices of 5 nearest neighbors for each point (excluding self)
    nearest_indices = np.argsort(distance_matrix, axis=1)[:,1:6]
    # Assign bonus values to just the nearest neighbors
    for i in range(n):
        nn_bonus[i, nearest_indices[i]] = np.exp(-0.1 * np.arange(5))
    
    noise = 1 + 0.05 * np.random.rand(*distance_matrix.shape)
    heuristic = noise * (0.7/normalized_dist + 0.3*nn_bonus)
    #EVOLVE-END       
    return heuristic