import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = np.exp(-np.linspace(0, 3, SearchAgents_no))  # Faster cooling
    w = np.square(Best_score) * rg  # Strengthened elite influence
    
    R1 = np.random.rand(SearchAgents_no, dim)
    R2 = np.random.rand(SearchAgents_no, dim)
    explore_prob = R1 * (1 - T.reshape(-1,1))  # Temp-modulated probability
    
    exploit_term = Best_pos + w*(Best_pos - Positions) * (1 + Best_score*rg)
    cp_idx = np.random.permutation(SearchAgents_no)
    explore_term = Positions[cp_idx] + T.reshape(-1,1)*(Positions - Positions.mean(0))
    
    Positions = np.where(explore_prob > 0.5, explore_term, exploit_term)
    #EVOLVE-END
    
    return Positions