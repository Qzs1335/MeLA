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
    # Enhanced opposition-based learning with adaptive probability
    prob = 0.4 + 0.2*np.sin(rg*np.pi/2)  # Dynamic probability
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < prob, opposite_pos, Positions)
    
    # Adaptive convergence with nonlinear decay
    t = 1 - (rg/2)**0.5  # Time-varying factor
    a = 2*(1 - t**2)
    r1 = np.random.randn(SearchAgents_no, dim)  # Gaussian noise
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Hybrid update rule
    A = (2*a*r1 - a)*t
    C = 2*r2*(1-t)
    D = np.abs(C*Best_pos - Positions) + 1e-8  # Avoid zero
    Positions = Best_pos - A*D + 0.1*(1-t)*np.random.randn(*Positions.shape)
    #EVOLVE-END       

    return Positions