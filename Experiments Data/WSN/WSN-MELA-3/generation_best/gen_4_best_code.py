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
    opp_prob = 0.4 + 0.3*np.exp(-5*rg)  # Nonlinear adaptive probability
    opposite_pos = lb_array + ub_array - Positions
    mask = np.random.rand(SearchAgents_no, dim) < opp_prob
    Positions = np.where(mask, opposite_pos * (0.9 + 0.2*np.random.rand()), Positions)
    
    # Hybrid convergence strategy
    t = 1 - rg
    a = 2 * (1 - t**3) * np.ones((SearchAgents_no, 1))
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Adaptive updates with exploration control
    A = (2*a*r1 - a) * (0.3 + 0.7*rg)
    C = 2*r2 * (0.5 + 0.5*t)
    D = np.abs(C*Best_pos - Positions)
    Positions = (1-t)*Best_pos + t*(Best_pos - A*D) + 0.2*t*np.random.randn(*Positions.shape)
    #EVOLVE-END       

    return Positions