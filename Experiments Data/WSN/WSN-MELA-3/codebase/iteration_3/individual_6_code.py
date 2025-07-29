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
    # Enhanced opposition with adaptive weight
    opp_weight = 0.7 * (1 - np.exp(-2*rg))  
    opposite_pos = lb_array + ub_array - Positions
    mask = np.random.rand(SearchAgents_no, dim) < opp_weight
    Positions = np.where(mask, opposite_pos*(1-0.3*rg) + Positions*0.3*rg, Positions)
    
    # Hybrid convergence strategy
    a = 2.5 * (1 - 0.7*rg) * np.linspace(1, 0, SearchAgents_no).reshape(-1,1)
    r1 = np.random.standard_cauchy((SearchAgents_no, dim))
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Adaptive position update
    A = np.clip((2*a*r1 - a), -3, 3)
    C = 2*r2 * (0.5 + 0.5*rg)
    D = np.abs(C*Best_pos - Positions)
    Positions = (Best_pos - A*D) * (0.9 + 0.1*rg) + (1-rg)*np.random.randn(*Positions.shape)
    #EVOLVE-END       

    return Positions