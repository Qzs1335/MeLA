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
    # Enhanced opposition-based learning
    opposite_pos = lb_array + ub_array - Positions
    use_opposite = np.random.rand(SearchAgents_no, dim) < (0.3 + 0.2*(1-rg))
    Positions = np.where(use_opposite, opposite_pos, Positions)
    
    # Adaptive convergence with perturbation
    a = 2 - 2*(np.log(np.linspace(1, np.exp(1), SearchAgents_no))).reshape(-1,1)
    r1 = np.random.randn(SearchAgents_no, dim)*0.1 + 1
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Hybrid update strategy
    A = (2*a*r1 - a)*(0.9 + 0.2*np.random.rand())
    C = 2*r2*(1 + 0.1*np.random.randn())
    D = np.abs(C*Best_pos - Positions)
    Positions = Best_pos - A*D + 0.1*rg*np.random.randn(SearchAgents_no, dim)
    #EVOLVE-END       

    return Positions