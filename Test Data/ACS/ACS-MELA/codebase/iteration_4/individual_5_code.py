import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    # The rest remains unchanged.    
    #EVOLVE-START
    SearchAgents_no, dim = Positions.shape  # Infer dimensions from input
    cos_p = np.cos(np.linspace(0, np.pi/2, SearchAgents_no))  # Smoother cooling curve
    T = (1 - Best_score) * cos_p.reshape(-1,1)  # Fitness-scaled temperature
    w = 1/(1 + np.exp(-rg))  # Non-linear weight adaptation
    
    R1 = np.random.rand(SearchAgents_no, dim)
    R2 = np.random.rand(SearchAgents_no, dim) * Best_score  
    
    explore_mask = (np.random.rand(SearchAgents_no, dim) > w)
    exploit_term = Best_pos + w * T * (R1 * Best_pos - R2 * Positions)
    explore_term = Positions + (R1 - R2) * np.random.permutation(Positions)
    
    Positions = np.where(explore_mask, explore_term, exploit_term)
    #EVOLVE-END       
    return Positions