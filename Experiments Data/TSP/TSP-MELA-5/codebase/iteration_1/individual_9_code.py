import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    # Convert to probabilities using softmax of negative distances
    # Add small constant to avoid numerical instability
    epsilon = 1e-10
    safe_distances = distance_matrix + epsilon
    
    # Use negative distances to make smaller distances more probable
    logits = -safe_distances
    
    # Compute softmax probabilities
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)  # Numerical stability
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    return probs
    #EVOLVE-END