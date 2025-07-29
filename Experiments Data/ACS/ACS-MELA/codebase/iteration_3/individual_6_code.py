import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = 1 - np.log(1 + np.linspace(0, 1, SearchAgents_no))  # Logarithmic cooling
    w = (1 - np.exp(-Best_score/rg))  # Sigmoid scaling
    
    R1 = np.random.randn(SearchAgents_no, dim) * 0.2  # Gaussian noise
    R2 = np.random.rand(SearchAgents_no, dim)  
    
    explore_mask = (np.random.rand(SearchAgents_no) > 0.5*(1-T)).reshape(-1,1)  # T-adaptive
    exploit_term = Best_pos + w * (R1 * (Best_pos - Positions))
    explore_term = Positions[np.random.permutation(SearchAgents_no)] * (1 + R2 - 0.5)
    
    Positions = np.where(explore_mask, explore_term, exploit_term)
    #EVOLVE-END       
    return Positions