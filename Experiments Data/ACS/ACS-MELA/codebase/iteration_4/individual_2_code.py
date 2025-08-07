import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape

    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array)) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = np.exp(-np.linspace(0, 5, SearchAgents_no))
    w = 1/(1+np.exp(-Best_score*rg/1000)) # Sigmoid scaling
    
    directional = np.cos(np.pi*(2/R1-1)) # Cosine perturbation
    R1 = np.random.rand(SearchAgents_no, dim)
    R2 = np.random.uniform(0.8,1.2, (SearchAgents_no,dim))
    
    explore_mask = (np.random.rand(SearchAgents_no) > 0.7)[:,None]
    exploit_term = Best_pos + w*(R1*directional)/(1+R2)
    
    cp_idx = np.random.permutation(SearchAgents_no)
    explore_term = Positions[cp_idx] + T[:,None]*(R1 - Best_pos[cp_idx])
    
    Positions = np.clip(np.where(explore_mask, explore_term, exploit_term), lb_array, ub_array)
    #EVOLVE-END
    
    return Positions