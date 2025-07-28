import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    # Compute inverse square distances with numerical stability
    inv_sq_dist = np.where(distance_matrix > 0, 
                          1 / (distance_matrix**2 + epsilon), 
                          0)
    # Compute log probabilities in stable way
    max_log = np.max(inv_sq_dist, axis=-1, keepdims=True)
    log_probs = inv_sq_dist - max_log
    # Compute softmax probabilities
    probs = np.exp(log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True)))
    # Ensure all probabilities are valid (non-negative, finite)
    probs = np.nan_to_num(probs, nan=1/probs.shape[-1])
    probs = np.clip(probs, 1e-10, 1.0)
    # Renormalize to ensure sum to 1
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return probs
    #EVOLVE-END