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
    opposite_pos = lb_array + ub_array - Positions
    opp_prob = 0.5 * (1 - np.exp(-rg))  # Decreases with iterations
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < opp_prob, opposite_pos, Positions)
    
    # Dimension-adaptive nonlinear convergence
    a = 2 * (1 - np.power(rg, 0.5)).reshape(-1, 1)
    r1 = np.random.randn(SearchAgents_no, dim)  # Normal distribution for better exploration
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Hybrid update with dynamic weights
    D = np.abs(2*r2*Best_pos - Positions)
    adaptive_weight = 0.5 + 0.5*np.sin(rg*np.pi/2)  # Oscillates for balance
    Positions = adaptive_weight*Best_pos + (1-adaptive_weight)*Positions - a*r1*D
    #EVOLVE-END       

    return Positions