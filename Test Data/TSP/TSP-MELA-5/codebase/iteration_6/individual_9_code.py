import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    # Add epsilon to avoid division by zero and ensure positive values
    safe_distances = distance_matrix + epsilon
    
    # Calculate stable weights using log transform
    log_weights = -2 * np.log(safe_distances)
    
    # Stable softmax implementation
    max_log = np.max(log_weights, axis=1, keepdims=True)
    exp_log = np.exp(log_weights - max_log)
    probs = exp_log / np.sum(exp_log, axis=1, keepdims=True)
    
    # Final epsilon to ensure no zeros
    probs = np.clip(probs, epsilon, 1.0)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    
    return probs
    #EVOLVE-END