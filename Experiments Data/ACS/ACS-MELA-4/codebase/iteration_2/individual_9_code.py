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
    # Improved chaotic map (logistic)
    chaos = 4.0 * rg * (1.0 - rg) * (1 - rg/2.0)
    
    # Dynamic opposition learning
    opp_prob = 0.5 * (1 + np.cos(np.pi * rg))
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < opp_prob, opposite_pos, Positions)
    
    # Elite-guided adaptive search
    w = chaos * (0.5 + 0.5 * np.random.rand(SearchAgents_no, 1))
    elite_mask = np.random.rand(SearchAgents_no, dim) < 0.7
    Positions = np.where(elite_mask, Best_pos * (1-w) + Positions * w, Positions)
    
    # Dimension-wise scaling
    scale = 0.9 + 0.2 * chaos * np.random.rand(SearchAgents_no, dim)
    Positions *= scale
    #EVOLVE-END
    
    return Positions