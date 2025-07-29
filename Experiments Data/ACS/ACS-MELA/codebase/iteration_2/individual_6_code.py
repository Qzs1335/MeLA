import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = 1/(1+np.linspace(0,5,SearchAgents_no))  # Sigmoid cooling
    w = np.clip(np.exp(-Best_score/10)*rg,0.1,0.9)  # Normalized weight
    
    R1 = np.random.normal(0.5,0.2,(SearchAgents_no,dim))
    R2 = np.random.standard_cauchy((SearchAgents_no,dim))
    
    explore_prob = 0.3 + 0.5*T.reshape(-1,1)  # Adaptive threshold
    explore_mask = np.random.rand(SearchAgents_no,dim) < explore_prob
    
    exploit_term = Best_pos + np.abs(w*R1*Best_pos) - np.abs(w*R2*Positions)
    cp_idx = np.random.choice(SearchAgents_no, size=SearchAgents_no, replace=False)
    explore_term = Positions[cp_idx] + T.reshape(-1,1)*(R1**2 - R2**2)
    
    Positions = np.where(explore_mask, explore_term/(1+T.reshape(-1,1)), exploit_term)
    #EVOLVE-END
    
    return Positions