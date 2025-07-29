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
    opp_prob = 0.5 * (1 - np.exp(-rg))  # Decreases with rg
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < opp_prob, opposite_pos, Positions)
    
    # Dynamic convergence factors
    t = 1 - (rg/2)  # Time-varying component
    a = 2*t - 2*t*np.random.rand(SearchAgents_no, 1)  # Random component per agent
    
    # Hybrid exploration-exploitation update
    exploit_mask = np.random.rand(SearchAgents_no, 1) < 0.5
    D_exploit = np.abs(Best_pos - Positions)
    D_explore = np.abs(Positions[np.random.permutation(SearchAgents_no)] - Positions)
    
    Positions = np.where(exploit_mask,
                       Best_pos - a*D_exploit,
                       Positions + a*D_explore)
    #EVOLVE-END       

    return Positions