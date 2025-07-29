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
    # Adaptive convergence factor
    t = 1 - (np.log(1 + (100 * rg)))
    w = 0.9 - (0.5 * t)
    
    # Opposition-based learning
    opposite_pos = lb_array + ub_array - Positions
    mask = np.random.rand(*Positions.shape) < 0.5
    Positions = np.where(mask, opposite_pos, Positions)
    
    # Memory-guided search
    memory = 0.5 * (Best_pos + Positions[np.random.permutation(SearchAgents_no)])
    Positions = w * Positions + (1-w) * (memory + np.random.randn(*Positions.shape) * rg)
    #EVOLVE-END
    
    return Positions