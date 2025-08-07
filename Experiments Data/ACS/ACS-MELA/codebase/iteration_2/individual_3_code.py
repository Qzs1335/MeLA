import numpy as np
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    * The rest remains unchanged. *
    #EVOLVE-START
    T = np.linspace(1, 0, SearchAgents_no) ** 2  # Enhanced cooling
    w = (0.1 + 0.9*T.reshape(-1,1)) * np.exp(-Best_score)  # Scaled adaptive weight
    
    R1 = np.random.randn(SearchAgents_no, dim)
    R2 = np.random.rand(SearchAgents_no, dim)
    
    p_explore = 0.3 * T.reshape(-1,1)  # Cooling-based probability
    explore_mask = np.random.rand(SearchAgents_no, dim) < p_explore
    
    exploit_term = Best_pos + w * (0.5*Best_pos*(1+R1) - Positions)
    cp_idx = np.random.permutation(SearchAgents_no)
    explore_term = Positions[cp_idx] + 0.5*(R1-R2)/(np.abs(R1-R2)+1e-8)
    
    Positions = np.where(explore_mask, explore_term, exploit_term)
    #EVOLVE-END    
    return Positions