import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = 1 - np.linspace(0, 1, SearchAgents_no)**0.5  # Enhanced cooling
    w = T * rg / (1 + Best_score)  # Score-adaptive weight
    
    R1 = np.random.rand(SearchAgents_no, dim)
    R2 = np.random.rand(SearchAgents_no, dim)
    
    explore_prob = 0.3 + 0.4 * T.reshape(-1, 1)  # Dynamic threshold
    exploit_term = Best_pos + w.reshape(-1,1) * (1-R1)*Best_pos - w.reshape(-1,1) * R2*Positions
    
    cp_idx = np.random.permutation(SearchAgents_no)
    explore_term = Positions[cp_idx] * (1 + T.reshape(-1,1)*(R1-R2))
    
    Positions = np.where(np.random.rand(SearchAgents_no,dim) < explore_prob, explore_term, exploit_term)
    Positions = np.clip(Positions, lb_array, ub_array)  # Boundary enforcement
    #EVOLVE-END
    
    return Positions