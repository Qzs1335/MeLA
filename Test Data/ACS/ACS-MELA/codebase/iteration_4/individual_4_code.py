import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = 1 - np.cos(np.linspace(0, np.pi/2, SearchAgents_no))  # Enhanced cooling
    t = np.tile(T.reshape(-1,1), (1,dim))
    w = (1 + Best_score*rg*0.01)/2  # Dual-factor weight
    
    R1 = np.random.rand(SearchAgents_no, dim)
    R2 = np.random.rand(SearchAgents_no, dim)
    
    explore_prob = 0.5 + 0.3*np.sin(np.linspace(0, np.pi, SearchAgents_no))
    explore_mask = np.tile(np.random.rand(SearchAgents_no,1)<explore_prob.reshape(-1,1), (1,dim))
    
    cp_idx = np.random.permutation(SearchAgents_no)
    explore_term = Positions[cp_idx] + t*(R1-R2) * np.random.normal(0,1,(SearchAgents_no,dim))
    exploit_term = Best_pos + w*(R1*Best_pos - R2*Positions)*np.cos(2*np.pi*R2) 
    
    Positions = np.where(explore_mask, explore_term, exploit_term)
    #EVOLVE-END
    
    return Positions