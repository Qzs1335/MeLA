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
    # Enhanced dynamic opposition
    opp_prob = 0.5*(1-np.exp(-2*rg))  # Faster decay
    opposite_pos = Best_pos + (lb_array + ub_array - Positions) * (1 - rg/2)
    use_opposite = np.random.rand(SearchAgents_no, dim) < opp_prob
    Positions = np.where(use_opposite, opposite_pos, Positions)
    
    # Nonlinear adaptive weights
    a = 2 * (1 - rg**2)
    r1 = np.random.randn(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Hybrid update
    A = (2*a*r1 - a) * (0.5 + rg*np.random.rand())
    C = 2*r2 * (0.5 + 0.5*rg)
    D = np.abs(C*Best_pos - Positions)
    Positions = Best_pos - A*D + (0.1*rg)*np.random.randn(*Positions.shape)
    #EVOLVE-END       

    return Positions