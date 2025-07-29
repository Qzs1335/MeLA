import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = np.exp(-np.linspace(0, 2, SearchAgents_no))  # Faster cooling
    w = 1/(1+np.exp(-Best_score)) * rg  # Sigmoid normalization
    
    R1 = np.random.randn(SearchAgents_no, dim)  # Gaussian noise
    R2 = np.random.rand(SearchAgents_no, dim)
    
    explore_mask = (np.random.rand(SearchAgents_no) > T).reshape(-1,1)  # Temp-driven exploration
    exploit_term = Best_pos * (1-w) + w*(R1*Best_pos + (1-R2)*Positions)
    
    cp_idx = np.random.permutation(SearchAgents_no)
    explore_term = Positions[cp_idx] * (1 + T.reshape(-1,1)*(R1-R2))
    
    Positions = np.where(explore_mask, explore_term, exploit_term)
    #EVOLVE-END
    
    return Positions