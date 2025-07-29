import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    # Convert distances to probabilities using softmax 
    # Add small constant to avoid division by zero
    epsilon = 1e-6
    inv_distances = 1 / (distance_matrix + epsilon)
    
    # Shift values to be centered around zero for numerical stability
    shift = inv_distances.max(keepdims=True)
    shifted_inv_distances = inv_distances - shift
    
    # Apply softmax for proper probability distribution
    exp_values = np.exp(shifted_inv_distances)
    probs = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
    
    # Clip tiny probabilities for safety
    probs = np.clip(probs, 1e-6, 1-1e-6)
    return probs
    #EVOLVE-END