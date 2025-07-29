import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START    
    epsilon = np.mean(distance_matrix)/1e4  # Dynamic epsilon
    w = np.tanh(1/(distance_matrix+epsilon))  # Adaptive weights
    hybrid = w*np.exp(1/(distance_matrix+epsilon)) + (1-w)/(distance_matrix+epsilon)
    return 1/(1 + np.exp(-hybrid))  # Sigmoid normalization          
    #EVOLVE-END       
    return hybrid