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
    # Adaptive opposition-based learning
    opp_prob = 0.3 + 0.4*(1 - rg/2)  # Range [0.3,0.7]
    opposite_pos = lb_array + ub_array - Positions
    mask = np.random.rand(SearchAgents_no, dim) < opp_prob
    Positions = np.where(mask, opposite_pos, Positions)
    
    # Hybrid convergence strategy
    t = 1 - (rg/2)**0.5  # Non-linear decay
    a = 2*(1 - t) + 0.1*t  # Smooth transition
    
    # Enhanced position updates
    r1 = np.random.randn(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    A = (2*a*r1 - a)*t
    C = 2*(1-t)*r2
    
    # Combined guidance
    D_best = np.abs(Best_pos - Positions)
    D_rand = np.abs(Positions[np.random.permutation(SearchAgents_no)] - Positions)
    Positions = Positions + (A*D_best + C*D_rand)/2
    #EVOLVE-END       

    return Positions