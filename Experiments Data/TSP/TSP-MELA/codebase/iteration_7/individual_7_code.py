import numpy as np
import numpy as np  

def heuristics_v2(distance_matrix):  
    #EVOLVE-START  
    try:  
        alpha = 1.2             # optimized pheromone weight  
        beta = 1.8              # tuned distance weight  
        stability_factor = 1e-12 # tighter stability threshold  

        distance_matrix = np.asarray(distance_matrix, dtype=np.float64)  
        if distance_matrix.size == 0:  
            raise ValueError("Empty distance matrix")  

        # Safeguard and prepare matrix  
        valid_mask = np.isfinite(distance_matrix)  
        distance_matrix = np.where(valid_mask, distance_matrix, np.inf)  
        distance_matrix[distance_matrix < stability_factor] = stability_factor  

        # Asymmetric logarithmic visibility  
        visibility = np.where(distance_matrix > 0,  
                            np.log1p(1/(distance_matrix + stability_factor)),  
                            -np.log1p(distance_matrix + stability_factor))  

        # Adaptive heuristic calculation  
        pheromone = np.ones_like(distance_matrix)  
        log_terms = alpha * np.log1p(pheromone) + beta * visibility  
        heuristic = np.exp(np.clip(log_terms, -100, 100))  # overflow protection  

        # Smart normalization  
        norm = np.linalg.norm(heuristic)  
        return heuristic / norm if norm > 0 else np.ones_like(heuristic)/heuristic.size  

    except Exception as e:  
        print(f"Heuristic Error: {str(e)}")  
        fallback = np.ones(distance_matrix.shape)/distance_matrix.size if hasattr(distance_matrix, 'shape') else None  
        return np.zeros_like(fallback) if fallback is None else fallback  
    #EVOLVE-END