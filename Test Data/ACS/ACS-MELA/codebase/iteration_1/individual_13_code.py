import numpy as np
import numpy as np 

def heuristics_v2(Positions, Best_pos, Best_score, rg):
    # Input validation
    Positions = np.array(Positions, dtype=float)
    Best_pos = np.array(Best_pos, dtype=float)
    if not isinstance(Best_score, (int, float)):
        raise ValueError("Best_score must be a numeric value")
    if not isinstance(rg, (int, float)):
        raise ValueError("rg must be a numeric value")

    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]

    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))

    # Boundary checking
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(*Positions.shape)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)

    #EVOLVE-START
    # Opposition-based learning component
    opposite_pos = 0.5*(lb_array + ub_array) + 0.5*(lb_array + ub_array - Positions)
    mask = np.random.rand(*Positions.shape) < 0.3
    Positions = np.where(mask, opposite_pos, Positions)
    
    # Adaptive dimension perturbation
    scale = 1 - np.exp(-0.1*(Best_score - Positions.mean(axis=1))[:, None])
    perturbation = scale * (np.random.randn(*Positions.shape) * 0.2 * abs(rg))
    new_pos = Best_pos + perturbation
    
    # Elite guidance selection
    elite_mask = (np.random.rand(SearchAgents_no) < 0.7)[:, None]
    Positions = np.where(elite_mask, new_pos, Positions)
    #EVOLVE-END
    
    # Final boundary check
    Positions = np.clip(Positions, lb_array, ub_array)
    
    return Positions