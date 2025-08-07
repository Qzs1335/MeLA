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
    # Enhanced opposition with adaptive range
    opp_range = 0.3 + 0.7*rg
    opp_pos = lb_array + opp_range*(ub_array - Positions)
    Positions = np.where(np.random.rand(SearchAgents_no,1) < 0.6-0.5*rg, opp_pos, Positions)
    
    # Nonlinear adaptive parameters
    a = 2*(1 - rg**2)
    r = np.random.randn(SearchAgents_no, dim)
    A = (2*a*np.random.rand() - a)*(0.7 + 0.3*np.sin(rg*np.pi))
    C = 1.5*(1 + np.random.rand())*rg
    
    # Hybrid update strategy
    D = np.abs(C*Best_pos - Positions)
    levy = 0.01*r/np.abs(r)**(1.5)
    Positions = Best_pos - A*D + levy*(1-rg)
    #EVOLVE-END       

    return Positions