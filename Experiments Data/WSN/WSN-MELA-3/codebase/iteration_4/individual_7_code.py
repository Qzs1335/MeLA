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
    # Improved dynamic opposition with adaptive noise
    opp_prob = 0.5*(1-np.exp(-2*rg))
    opp_noise = 0.1*rg*np.random.randn(*Positions.shape)
    Positions = np.where(np.random.rand(SearchAgents_no,dim)<opp_prob, 
                       lb_array + ub_array - Positions + opp_noise, 
                       Positions)
    
    # Nonlinear convergence with hybrid updates
    a = 2*(1 - rg**2)
    r = np.random.rand(SearchAgents_no,1)
    A = (2*a*r - a)*(0.8 + 0.4*np.random.rand())
    C = 2*(1 + np.sin(rg*np.pi/2))*np.random.rand(*Positions.shape)
    D = np.abs(C*Best_pos - Positions)
    Positions = Best_pos - A*D + 0.2*rg*np.random.randn(*Positions.shape)
    #EVOLVE-END       

    return Positions