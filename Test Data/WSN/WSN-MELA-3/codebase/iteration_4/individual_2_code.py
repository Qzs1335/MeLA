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
    opp_prob = 0.5 * (1 - np.exp(-5*rg))  # Faster decay
    opposite_pos = lb_array + ub_array - Positions
    mask = np.random.rand(SearchAgents_no, dim) < opp_prob
    Positions = np.where(mask, opposite_pos*(0.9 + 0.2*np.random.rand()), Positions)
    
    # Nonlinear convergence factors
    a = 2 * (1 - np.tanh(rg*np.linspace(0,3,SearchAgents_no))).reshape(-1,1)
    r1 = np.random.weibull(1.5, (SearchAgents_no, dim))
    r2 = np.random.power(3, (SearchAgents_no, dim))
    
    # Hybrid update strategy
    A = (2*a*r1 - a) * (0.3 + 0.7*rg)  # Adaptive scaling
    C = 2*r2 * (0.1 + 0.9*(1-rg)**2)
    D = np.abs(C*Best_pos - Positions)
    Positions = Best_pos - A*D + (0.05 + 0.15*rg)*np.random.randn(*Positions.shape)
    #EVOLVE-END       

    return Positions