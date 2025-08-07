import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = np.exp(-np.linspace(0, 5, SearchAgents_no)**0.5)  # Steeper cooling  
    w = np.exp(-Best_score) * rg * (0.3 + np.log1p(dim))  # Dimensional scaling
    
    R1 = rg * np.random.randn(SearchAgents_no, dim)  # Levy-like steps
    R2 = np.random.rand(SearchAgents_no, dim)
    
    explore_mask = np.random.rand(*Positions.shape).argsort(axis=1) < int(dim*0.7)
    exploit_term = Best_pos + w * np.clip(R1, -2, 2) * (Best_pos - Positions)
    
    cp_idx = np.random.permutation(SearchAgents_no)
    explore_term = Positions[cp_idx] * (1 + T.reshape(-1,1)*(R1-R2)/dim)
    
    Positions = np.where(explore_mask, explore_term, exploit_term)
    #EVOLVE-END
    
    return Positions