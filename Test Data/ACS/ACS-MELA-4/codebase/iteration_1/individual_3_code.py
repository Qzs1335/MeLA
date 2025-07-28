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
    # Chaotic map for better randomness
    chaos = 4 * rg * (1 - rg)
    
    # Opposition-based learning
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < 0.5, opposite_pos, Positions)
    
    # Adaptive cosine search
    t = np.pi * np.random.rand(SearchAgents_no, 1)
    w = 0.5 * (1 + np.cos(t)) * chaos
    new_pos = Best_pos * (1 - w) + Positions * w
    
    # Dynamic scaling
    scale = 1 - (np.random.rand(SearchAgents_no, dim) * chaos)
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < 0.5, new_pos * scale, new_pos)
    #EVOLVE-END
    
    return Positions