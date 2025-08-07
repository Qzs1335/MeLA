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
    opp_prob = 0.6*(1 - np.exp(-2*rg))  # Faster adaptation
    opposite_pos = lb_array + ub_array - Positions
    mask = (opposite_pos >= lb_array) & (opposite_pos <= ub_array)
    Positions = np.where((np.random.rand(*Positions.shape) < opp_prob) & mask, opposite_pos, Positions)
    
    # Nonlinear convergence control
    a = 2 - 2*(rg**0.5)*np.linspace(0, 1, SearchAgents_no).reshape(-1,1)
    r1 = np.random.rand(*Positions.shape)
    r2 = np.random.rand(*Positions.shape)
    
    # Hybrid update strategy
    A = (2*a*r1 - a)*(0.3 + 0.7*rg)
    C = 2*r2*(0.5 + 0.5*rg)
    D = np.abs(C*Best_pos - Positions)
    Positions = (1-rg)*Best_pos - A*D + rg*np.random.randn(*Positions.shape)
    #EVOLVE-END       

    return Positions