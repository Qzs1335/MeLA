import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = np.exp(-np.linspace(0, 3, SearchAgents_no)) ** 2  # Faster cooling
    w = 1/(1+np.exp(-Best_score/1000)) * rg  # Sigmoid weight scaling
    
    R = np.random.randn(SearchAgents_no, dim)  # Normal distribution for better diversity
    
    explore_mask = (np.random.rand(SearchAgents_no, 1) > T.reshape(-1,1))  # Dynamic threshold
    exploit_term = Best_pos + w * (Best_pos - Positions) / (1 + R**2)
    
    cp_idx = np.random.permutation(SearchAgents_no)
    explore_term = Positions[cp_idx] + T.reshape(-1,1) * R
    
    Positions = np.where(explore_mask, explore_term, exploit_term)
    #EVOLVE-END
    
    return Positions