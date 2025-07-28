import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(SearchAgents_no, dim)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    T = 1/(1+np.exp(np.linspace(0,3,SearchAgents_no))) # Sigmoid cooling
    w = 0.1 + 0.9/(1+np.exp(-Best_score)) # Normalized adaptive weight
    
    R = np.random.randn(SearchAgents_no, dim) * rg/10 # Scaled noise
    cp_idx = np.random.permutation(SearchAgents_no)
    
    grad = (Best_pos - Positions) * w 
    diversify = (Positions[cp_idx] - Positions) * T[:,None]
    
    Positions += np.where(np.random.rand(SearchAgents_no,1)<0.5, grad, diversify) + R
    #EVOLVE-END
    
    return Positions