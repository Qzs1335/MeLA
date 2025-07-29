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
    # Enhanced opposition learning
    opp_prob = 0.5 / (1 + np.exp(5*(rg-0.5)))
    opposite_pos = lb_array + ub_array - Positions
    mask = (np.random.rand(SearchAgents_no, dim) < opp_prob)
    Positions = np.where(mask, opposite_pos*(0.9 + 0.2*np.random.rand()), Positions)
    
    # Nonlinear convergence
    a = 2 / (1 + np.exp(3*rg*np.linspace(0,1,SearchAgents_no))).reshape(-1,1)
    r1, r2 = np.random.rand(2, SearchAgents_no, dim)
    
    # Hybrid update
    A = (2*a*r1 - a) * (0.3 + 0.7*rg)
    C = 2*r2 * (0.1 + 0.9*np.sqrt(rg))
    D = np.abs(C*Best_pos - Positions)
    Positions = (1-rg)*Best_pos + rg*Positions - A*D + 0.05*np.exp(-2*rg)*np.random.randn(*Positions.shape)
    #EVOLVE-END       

    return Positions