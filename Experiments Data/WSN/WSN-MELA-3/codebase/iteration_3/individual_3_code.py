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
    # Enhanced opposition-based learning
    opp_prob = 0.7 * (1 - np.exp(-3*rg))  # Faster decay
    opposite_pos = lb_array + ub_array - Positions
    hybrid_pos = Positions + 0.5*(opposite_pos - Positions)*np.random.rand(*Positions.shape)
    mask = np.random.rand(SearchAgents_no, dim) < opp_prob
    Positions = np.where(mask, hybrid_pos, Positions)
    
    # Nonlinear adaptive factors
    t = np.linspace(0, 1, SearchAgents_no).reshape(-1,1)
    a = 2*(1 - (rg**0.5)*t**2)
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Hybrid update strategy
    A = (2*a*r1 - a)
    C = 2*r2*(1 - rg**2)
    D = np.abs(C*Best_pos - Positions)
    cos_factor = np.cos(2*np.pi*np.random.rand(SearchAgents_no, dim))
    Positions = Best_pos - A*D*cos_factor + (0.2*rg)*np.random.randn(*Positions.shape)
    #EVOLVE-END       

    return Positions