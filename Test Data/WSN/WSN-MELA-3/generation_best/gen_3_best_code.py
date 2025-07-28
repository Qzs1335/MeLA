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
    # Enhanced dynamic opposition with sigmoid probability
    opp_prob = 1/(1 + np.exp(3*rg-1.5)) 
    opposite_pos = lb_array + ub_array - Positions
    mask = np.random.rand(SearchAgents_no, dim) < opp_prob
    Positions = np.where(mask, opposite_pos, 0.9*Positions + 0.1*opposite_pos)
    
    # Non-linear adaptive convergence
    t = 1 - rg
    a = 2 * (1 - t**3)
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Hybrid update strategy
    A = (2*a*r1 - a) * (0.3 + 0.7*np.random.rand())
    C = 2*r2 * (0.5 + 0.5*t)
    D = np.abs(C*Best_pos - Positions)
    Positions = Best_pos - A*D + t*np.random.randn(*Positions.shape)*(ub_array-lb_array)/10
    #EVOLVE-END       

    return Positions