import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = 1/(1 + np.linspace(0, 10, SearchAgents_no))  # Enhanced cooling
    w = np.exp(-1/abs(Best_score+1e-9)) * rg/2  # Normalized weight
    
    R1 = np.random.standard_cauchy((SearchAgents_no, dim)) * 0.1
    R2 = np.random.normal(0, 0.3, (SearchAgents_no, dim))
    
    explore_mask = np.random.rand(*Positions.shape) > (0.3 + 0.5*T.reshape(-1,1))
    
    delta_pos = Best_pos - Positions
    exploit_term = Best_pos + w * (R1 * delta_pos - R2 * Positions)
    
    cp_idx = np.random.choice(SearchAgents_no, size=SearchAgents_no, replace=False)
    explore_term = Positions[cp_idx] + np.abs(R1) * (Positions - Positions[cp_idx]) * T.reshape(-1,1)
    
    Positions = np.where(explore_mask, explore_term, exploit_term)
    #EVOLVE-END
    
    return Positions