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
    # Enhanced chaotic map
    chaos = 4 * rg * (1 - rg) * (1 - 0.5*np.random.rand())
    
    # Dynamic opposition probability
    opp_prob = 0.5 * (1 + np.cos(np.pi*rg))
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < opp_prob, opposite_pos, Positions)
    
    # Elite-guided cosine search
    t = np.pi * np.random.rand(SearchAgents_no, 1)
    w = (0.3 + 0.7*rg) * (1 + np.cos(t)) * chaos
    new_pos = Best_pos * (1 - w) + Positions * w
    
    # Non-linear scaling
    scale = np.exp(-chaos * np.random.rand(SearchAgents_no, dim))
    Positions = np.where(scale < 0.7, new_pos * scale, new_pos)
    #EVOLVE-END
    
    return Positions