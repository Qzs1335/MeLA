import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = np.linspace(0, 1, SearchAgents_no)**2  # Quadratic cooling
    w = 0.5 * (1 + np.cos(np.pi * Best_score))  # Cosine-adapted weight
    
    R1 = np.random.standard_cauchy((SearchAgents_no, dim))
    R2 = np.random.rand(SearchAgents_no, dim)
    
    explore_mask = np.random.rand(SearchAgents_no,dim) > (1-T).reshape(-1,1)  # Annealed threshold
    exploit_term = Best_pos + w*(np.abs(R1)*Best_pos - R2*Positions)
    
    cp_idx = np.random.permutation(SearchAgents_no)
    explore_term = Positions[cp_idx] * (1 + 0.1*R1) 
    #EVOLVE-END  
    
    return Positions