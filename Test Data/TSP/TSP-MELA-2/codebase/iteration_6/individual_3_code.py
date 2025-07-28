import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-10
    top_k = np.sort(distance_matrix, axis=1)[:,1:5]
    weights = np.mean(top_k, axis=1)
    temperature = 0.5 * np.max(distance_matrix)
    # Define sigmoid function locally
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    return np.exp(-(distance_matrix**2)/(weights[:,None]*temperature))/((distance_matrix + eps)*sigmoid(weights[:,None]/temperature))
    #EVOLVE-END
    return 1 / distance_matrix