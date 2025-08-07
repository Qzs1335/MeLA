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
    t = (np.arange(1, SearchAgents_no+1) % 10) + 1
    cos_factor = np.cos(np.pi * t / 10).reshape(-1,1)
    levy = rg * np.random.randn(*Positions.shape) / np.power(np.abs(np.random.randn(*Positions.shape)), 1/1.5)
    
    exploit_mask = (np.random.rand(SearchAgents_no,1) < 0.7 + 0.3*cos_factor)
    neighborhood = Positions + 0.04*np.random.randn(*Positions.shape)*(Best_pos - Positions)
    diversity = 0.1 * levy * (Positions.mean(0) - Positions)
    
    Positions = np.where(exploit_mask, 
                         (1-cos_factor)*neighborhood + cos_factor*Best_pos, 
                         Positions + diversity)
    #EVOLVE-END
    
    return Positions