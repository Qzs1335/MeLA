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
    chaos = 4 * rg * (1 - rg) * (3.9 - 0.9*rg) 
    
    # Adaptive opposition learning
    opp_threshold = 0.5*(1 + np.cos(np.pi*rg))
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < opp_threshold, opposite_pos, Positions)
    
    # Elite-guided cosine search
    elite = Positions[np.argsort(Best_score)[:3]].mean(axis=0)
    t = np.pi * np.random.rand(SearchAgents_no, 1)
    w = (0.9 - 0.4*rg) * (1 + np.cos(t)) * chaos
    new_pos = elite*(1-w) + Positions*w
    
    # Dimension-aware scaling
    scale = (0.1 + 0.9*rg) * (1 - np.random.rand(SearchAgents_no, dim)*chaos/np.sqrt(dim))
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < 0.7*rg, new_pos*scale, new_pos)
    #EVOLVE-END
    
    return Positions