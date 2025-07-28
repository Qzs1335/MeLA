import numpy as np
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = np.exp(-np.linspace(0, 5, SearchAgents_no)) * rg  # Temperature-scaled cooling
    w = 0.5*(1-np.exp(-Best_score/rg))  # Sigmoid-normalized weight
    
    cp_idx = np.random.permutation(SearchAgents_no)
    R = np.random.randn(SearchAgents_no, dim) * T.reshape(-1,1)
    
    exploit_term = Best_pos + w*(Best_pos - Positions)
    Positions = np.where(R > 0, Positions[cp_idx] + R, exploit_term)
    #EVOLVE-END
    
    return Positions