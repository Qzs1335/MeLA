import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        distance_matrix = np.asarray(distance_matrix)
        assert distance_matrix.ndim == 2 and distance_matrix.shape[0] == distance_matrix.shape[1], "Non-square matrix"
        
        np.fill_diagonal(distance_matrix, np.inf)
        visibility = np.reciprocal(distance_matrix, where=distance_matrix>0)
        
        # Apply softmax for probabilistic interpretation
        max_v = np.max(visibility, axis=1, keepdims=True)
        exp_v = np.exp(visibility - max_v)  # Numerically stable
        normalized = exp_v / np.sum(exp_v, axis=1, keepdims=True)
        
        return normalized
        
    except Exception as e:
        print(f"Heuristic computation error: {str(e)}")
        size = distance_matrix.shape[0] if isinstance(distance_matrix, np.ndarray) else len(distance_matrix)
        return np.eye(size) if size > 0 else None
    #EVOLVE-END