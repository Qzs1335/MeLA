import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(*Positions.shape)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    # Adaptive OBL with cosine weighting - fixed broadcasting
    max_pos = np.max(Positions, axis=1, keepdims=True) + 1e-12
    dyn_w = 0.9 * (0.5 * (1 + np.cos(np.pi * Positions / max_pos)))
    Positions = dyn_w * Positions + (1 - dyn_w) * np.random.uniform(0, 0.5, Positions.shape)

    # Gradient-aware elite hybrid - fixed weight dimensions
    elite_weights = np.exp(-0.5 * np.mean(np.abs(Positions - Best_pos), axis=1, keepdims=True))
    permutation = Positions[np.random.permutation(SearchAgents_no)]
    elite_strategy = Best_pos + rg * (permutation - Positions)
    Positions = 0.5 * (Positions + elite_weights * elite_strategy)

    # Score-adaptive mutation - fixed probability dimensions
    pos_range = np.max(Positions) - np.min(Positions)
    min_prob = 0.1
    mutation_prob = min_prob + (1 - min_prob) * (Best_score / (Best_score + pos_range))
    mutation_prob = np.clip(mutation_prob, min_prob, 1.0)
    mut_mask = np.random.rand(*Positions.shape) < mutation_prob
    delta = np.tan(np.pi * (np.random.rand(*Positions.shape) - 0.1))
    Positions = np.where(mut_mask, np.abs(Positions + delta * 0.1), Positions)
    #EVOLVE-END       
    
    return Positions