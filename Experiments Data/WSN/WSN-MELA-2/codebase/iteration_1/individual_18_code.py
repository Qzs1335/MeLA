import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(*Positions.shape)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    adaptive_factor = 0.5 * (1 + np.sin(0.1 * np.arange(SearchAgents_no)).reshape(-1,1))
    memory = 0.3 * np.random.rand(SearchAgents_no, dim) * Best_pos
    turbulence = 0.1 * rg * (np.random.rand(SearchAgents_no, dim) - 0.5)
    
    Positions = adaptive_factor * (Positions + memory) + (1 - adaptive_factor) * Best_pos + turbulence
    #EVOLVE-END
    
    return Positions