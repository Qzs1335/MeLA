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
    # Opposition-based learning
    opposite_pos = lb_array + ub_array - Positions
    candidate_pos = np.where(np.random.rand(SearchAgents_no, dim) < 0.5, Positions, opposite_pos)
    
    # Chaotic search using logistic map
    chaos = 3.9 * np.random.rand(SearchAgents_no, dim) * (1 - np.random.rand(SearchAgents_no, dim))
    
    # Dynamic parameter adjustment
    adaptive_rg = rg * (1 - np.exp(-0.1 * Best_score))
    R = adaptive_rg * (2 * np.random.rand(SearchAgents_no, 1) - 1)
    
    # Hybrid update
    exploit_mask = np.abs(R) <= 1
    Positions = np.where(exploit_mask.reshape(-1,1),
                       Best_pos + chaos * (Best_pos - candidate_pos),
                       candidate_pos + chaos * (candidate_pos - Positions))
    #EVOLVE-END       
    return Positions