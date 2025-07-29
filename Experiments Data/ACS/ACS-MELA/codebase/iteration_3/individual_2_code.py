import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = 0.1 + 0.9*np.exp(-np.linspace(0, 3, SearchAgents_no))  # Smoother cooling
    w = 1/(1+np.exp(-Best_score))  # Sigmoid normalization
    
    R1 = np.random.randn(SearchAgents_no, dim) 
    R2 = np.random.rand(SearchAgents_no, dim)
    V = w*(Best_pos - Positions)  # Velocity vector
    
    cp_idx = np.random.permutation(SearchAgents_no)
    explore_term = Positions + T.reshape(-1,1)*(Positions[cp_idx] - Positions + 0.1*R1)
    exploit_term = Positions + V + 0.1*R2*np.abs(V)
    
    mask = np.random.rand(SearchAgents_no, 1) < T.reshape(-1,1) 
    Positions = np.where(mask, explore_term, exploit_term)
    #EVOLVE-END
    
    return Positions