import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = 1/(1+np.linspace(0,10,SearchAgents_no))    # More aggressive cooling
    w = 0.1 + 0.9*np.exp(-Best_score/rg)          # Normalized adaptive weight
    
    R1 = rg * np.random.randn(SearchAgents_no, dim)  
    R2 = np.random.rand(SearchAgents_no, dim)
    
    explore_mask = np.random.rand(SearchAgents_no, dim) < T.reshape(-1,1)
    exploit_term = Best_pos + w*(R1*Best_pos - R2*Positions)
    
    cp_idx = np.random.permutation(SearchAgents_no)
    explore_term = Positions + T.reshape(-1,1) * (Positions[cp_idx] - Positions)
    
    Positions = np.where(explore_mask, explore_term, exploit_term)
    #EVOLVE-END
    
    return Positions