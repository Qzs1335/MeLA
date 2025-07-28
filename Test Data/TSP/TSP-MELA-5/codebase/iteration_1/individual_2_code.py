import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    # Ensure distance matrix has positive values
    distance_matrix = np.abs(distance_matrix)
    # Add small constant to avoid log(0)
    epsilon = 1e-10
    # Apply safe transformation
    transformed = np.log(1 / (distance_matrix + epsilon))
    # Clip extreme values to prevent overflow
    transformed = np.clip(transformed, -100, 100)
    # Take square root while preserving sign
    result = np.sqrt(np.abs(transformed)) * np.sign(transformed)
    # Convert to probabilities using softmax
    exp_values = np.exp(result - np.max(result, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities
    #EVOLVE-END