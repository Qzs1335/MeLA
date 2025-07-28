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
    # Dynamic opposition-based learning
    dyn_prob = 0.3 + 0.4*(1 - rg)
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < dyn_prob, opposite_pos, Positions)
    
    # Adaptive nonlinear convergence
    a = 2 - (2 - 0.5*rg)*(np.log(np.linspace(1, np.exp(1), SearchAgents_no))).reshape(-1, 1)
    r1 = np.random.randn(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Hybrid position update
    A = (2*a*r1 - a)*np.clip(rg, 0.1, 1)
    C = 2*r2
    D_leader = np.abs(C*Best_pos - Positions)
    D_random = np.abs(C*Positions[np.random.permutation(SearchAgents_no)] - Positions)
    Positions = np.where(np.random.rand(SearchAgents_no,1)<0.7, 
                        Best_pos - A*D_leader,
                        Positions + A*D_random)
    #EVOLVE-END       

    return Positions