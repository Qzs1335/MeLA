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
    # Opposition-based learning (within same agent count)
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < 0.5, opposite_pos, Positions)
    
    # Nonlinear convergence factor properly shaped
    a = 2 - 2*(np.log(np.linspace(1, np.exp(1), SearchAgents_no))).reshape(-1, 1)
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Adaptive position update with proper broadcasting
    A = 2*a*r1 - a
    C = 2*r2
    D = np.abs(C*Best_pos.reshape(1, -1) - Positions)
    Positions = Best_pos.reshape(1, -1) - A*D
    #EVOLVE-END       

    return Positions