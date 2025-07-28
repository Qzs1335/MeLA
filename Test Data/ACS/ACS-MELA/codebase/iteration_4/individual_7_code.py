import numpy as np
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    * The rest remains unchanged. *
    #EVOLVE-START
    T = 1/(1 + np.linspace(0, 5, SearchAgents_no)**0.5)  # Square-root cooling for better exploration 
    w = 0.1 + 0.9*np.exp(-Best_score)  # Smooth dynamic weighting
    
    delta = Best_pos - Positions
    cos_sim = np.exp(-np.linalg.norm(delta, axis=1)).reshape(-1,1)  # Similarity-based coefficient
    
    R1 = np.random.randn(SearchAgents_no, dim)
    R2 = np.random.randn(SearchAgents_no, dim)
    
    explore_mask = np.random.rand(SearchAgents_no, dim) > (0.4 + 0.5*cos_sim)  # Adaptive mask
    cp_idx = np.random.permutation(SearchAgents_no)
    explore_term = Positions[cp_idx] + T.reshape(-1,1)*(R1 - R2)
    exploit_term = Best_pos + w*cos_sim*(R1*Best_pos - R2*Positions)
    
    Positions = np.where(explore_mask, explore_term, exploit_term)
    #EVOLVE-END
    return Positions