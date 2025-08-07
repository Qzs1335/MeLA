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
    opposite_pos = lb_array + ub_array - Positions
    mask = (np.random.rand(*Positions.shape) < opp_prob*(1-0.5*rg))
    Positions = np.where(mask, opposite_pos + 0.1*rg*np.random.randn(*Positions.shape), Positions)
    
    # Nonlinear adaptive updates
    a = 2*(1 - np.tanh(rg*np.linspace(0,1,SearchAgents_no).reshape(-1,1)))
    r1, r2 = np.random.rand(2, SearchAgents_no, dim)
    A = (2*a*r1 - a)*(0.7 + 0.3*np.random.rand())
    C = 2*r2*(1 - rg**2)  # Quadratic decay
    D = np.abs(C*Best_pos - Positions)
    Positions = Best_pos - A*D + 0.2*rg*np.random.randn(*Positions.shape)
    #EVOLVE-END       

    return Positions
