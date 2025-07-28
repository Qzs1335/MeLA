import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions) 

    #EVOLVE-START
    T = 1/(1 + np.exp(np.linspace(0, 5, SearchAgents_no)))  # Sigmoid cooling
    w = 0.5 + rg * (1 - np.exp(-Best_score*0.1))  # Bounded adaptive weight
    
    R1 = np.random.standard_cauchy((SearchAgents_no, dim)) * rg  # Heavy-tailed
    R2 = np.random.normal(0, 1, (SearchAgents_no, dim))
    
    explore_mask = np.random.rand(SearchAgents_no, dim) > T[:, None]
    exploit_term = Best_pos + w*(Best_pos - Positions)/(1 + R2)
    
    cp_idx = np.random.permutation(SearchAgents_no)
    explore_term = Positions[cp_idx] * (1 + R1) 
    
    Positions = np.where(explore_mask, explore_term, exploit_term)
    Positions = np.clip(Positions, lb_array, ub_array)
    #EVOLVE-END
    
    return Positions