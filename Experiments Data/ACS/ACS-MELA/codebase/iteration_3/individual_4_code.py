import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = 0.7 * np.exp(-np.linspace(0, 3, SearchAgents_no))  # Smoother cooling
    w = 1.5 / (1 + Best_score) * rg  # Normalized weight
    
    R = np.random.randn(SearchAgents_no, dim)  # Gaussian noise
    cp_idx = np.random.permutation(SearchAgents_no)
    
    exploit = Best_pos + w * (Positions - Best_pos) * R
    explore = Positions[cp_idx] * (1 + R * T.reshape(-1,1))
    
    Positions = np.where(R > 0, exploit, explore)  # Variance-based branching
    #EVOLVE-END
    
    return Positions