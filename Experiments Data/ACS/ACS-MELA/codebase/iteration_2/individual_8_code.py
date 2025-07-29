import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = np.exp(-np.linspace(0, 5, dim))  # Dimension-wise cooling  
    w = (1 - np.exp(-Best_score)) * T * rg  # Unified adaptive factor
    
    R = np.random.rand(SearchAgents_no, dim)
    explore_mask = (np.random.rand(dim) > 0.5)  # Dimension-specific mask
    
    exploit_term = Best_pos + w * (R * Best_pos - Positions) 
    cp_idx = np.random.permutation(SearchAgents_no)
    explore_term = Positions[cp_idx] + T * (R - 0.5)
    
    Positions = np.where(explore_mask[np.newaxis,:], explore_term, exploit_term)
    #EVOLVE-END
    
    return Positions