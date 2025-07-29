import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    # Convert distances to similarities (small distances -> high similarity)
    epsilon = 1e-10
    similarities = 1 / (distance_matrix + epsilon)
    
    # Apply softmax along each row to get valid probabilities
    max_sim = np.max(similarities, axis=1, keepdims=True)
    exp_sim = np.exp(similarities - max_sim)  # Numerically stable
    probabilities = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
    
    # Verify sum to 1 (debug check, optional)
    # np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(probabilities.shape[0]), rtol=1e-6)
    
    return probabilities
    #EVOLVE-END