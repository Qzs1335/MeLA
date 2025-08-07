import numpy as np
import numpy as np

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    try:
        beta = 2  # Pure distance-based influence
        
        # Input processing with diagonal cleanup
        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Logarithmic attenuation of distances
        scores = -beta * np.log(distance_matrix + 1e-10)
        
        # Softmax normalization for probabilistic weights
        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)  # Numerical stability
        return exp_scores / np.sum(exp_scores)
        
    except Exception as e:
        print(f"Heuristic calculation error: {str(e)}")
        size = distance_matrix.shape[0] if hasattr(distance_matrix, 'shape') else 1
        return np.ones(size)/size  # Uniform fallback
    #EVOLVE-END