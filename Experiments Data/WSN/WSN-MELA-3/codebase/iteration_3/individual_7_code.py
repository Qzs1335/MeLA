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
    # Dynamic opposition with adaptive scaling
    opp_scale = 0.3 + 0.7/(1+np.exp(-5*(rg-0.5)))
    opposite_pos = lb_array + ub_array - opp_scale*Positions
    opp_mask = np.random.rand(SearchAgents_no, dim) < 0.6*np.exp(-rg)
    Positions = np.where(opp_mask, opposite_pos, Positions)
    
    # Sigmoid-adapted convergence
    a = 2 / (1 + np.exp(3*(rg-0.5)))
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Hybrid update with local search
    A = (2*a*r1 - a) * (0.7 + 0.3*np.random.rand())
    C = 2*r2 * (0.1 + 0.9*(1-rg))
    D = np.abs(C*Best_pos - Positions)
    local_search = 0.2*(1-rg)*np.random.randn(*Positions.shape)
    Positions = Best_pos - A*D + local_search
    #EVOLVE-END       

    return Positions