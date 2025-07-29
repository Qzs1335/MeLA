import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = 1 - np.linspace(0,1,SearchAgents_no)**2  # Quadratic cooling
    w = (0.5 + np.exp(-abs(Best_score))) * rg  # Dynamic weight
    
    R = np.random.permutation(np.logspace(-3,0,dim))[None,:] # Directional diversity
    
    # Dual-phase update
    explore_term = np.roll(Positions, shift=1, axis=0) * (1 + T.reshape(-1,1)*R)
    exploit_term = Best_pos * w + Positions * (1-w) * (1 - 0.5*T.reshape(-1,1))
    
    phase_mask = (T > 0.3).reshape(-1,1) # Early-stage exploration
    Positions = np.where(phase_mask, explore_term, exploit_term)
    #EVOLVE-END
    
    return Positions