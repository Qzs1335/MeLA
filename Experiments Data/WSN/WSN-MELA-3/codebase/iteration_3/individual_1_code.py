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
    # Enhanced opposition with nonlinear probability
    opp_prob = 0.7 * (1 - np.exp(-2*rg))  
    opp_mask = np.random.rand(SearchAgents_no, dim) < opp_prob
    Positions = np.where(opp_mask, lb_array + ub_array - Positions, Positions)
    
    # Adaptive convergence with nonlinear scaling
    a = 3 * (1 - np.tanh(rg * np.linspace(0.5, 1.5, SearchAgents_no))).reshape(-1, 1)
    r1 = np.random.standard_cauchy((SearchAgents_no, dim)) * 0.1
    r2 = np.random.power(3, (SearchAgents_no, dim))
    
    # Hybrid update strategy
    A = (2*a*r1 - a) * (0.3 + 0.7*rg)
    C = 3*r2 * np.sqrt(1 - rg)
    D = np.abs(C*Best_pos - Positions)
    Positions = (1-rg)*Positions + rg*(Best_pos - A*D) + 0.2*(1-rg)**2*np.random.randn(*Positions.shape)
    #EVOLVE-END       

    return Positions