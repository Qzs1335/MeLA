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
    # Enhanced dynamic opposition with nonlinear scaling
    opp_scale = 0.3 + 0.7*np.exp(-5*rg)
    opposite_pos = lb_array + ub_array - opp_scale*Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < 0.6*np.tanh(3*rg), opposite_pos, Positions)
    
    # Hybrid convergence strategy
    a = 2 - 2*(1 + np.cos(np.pi*rg/2))**2
    r1 = np.random.weibull(1.5, (SearchAgents_no, dim))
    r2 = np.random.power(3, (SearchAgents_no, dim))
    
    # Balanced update equation
    A = (2*a*r1 - a) * (0.3 + 0.7*rg)
    C = 2*r2 * (0.1 + 0.9*rg**0.5)
    D = np.abs(C*Best_pos - Positions)
    Positions = (1-rg)*Best_pos - A*D + rg*np.random.randn(*Positions.shape)
    #EVOLVE-END       

    return Positions