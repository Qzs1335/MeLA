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
    b = 1  # Spiral constant
    l = np.random.uniform(-1, 1, (SearchAgents_no, 1))
    a = 2 - 2 * (rg/100)  # Adaptive parameter
    
    D = np.abs(Best_pos - Positions)
    spiral_update = D * np.exp(b * l) * np.cos(2 * np.pi * l) * a
    random_walk = 0.5 * (Best_pos - Positions) * np.random.rand(*Positions.shape)
    
    exploit = np.random.rand(SearchAgents_no, 1) < 0.5
    Positions = np.where(exploit, Best_pos + spiral_update, Positions + random_walk)
    #EVOLVE-END       
    
    return Positions