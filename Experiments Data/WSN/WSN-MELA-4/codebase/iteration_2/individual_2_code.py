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
    # Enhanced adaptive parameters
    progress = np.clip(1 - (Best_score/1000), 0.1, 0.9)
    w = 0.3 + 0.5 * progress
    c1 = 1.5 * (1-progress)
    c2 = 1.5 + 0.5*progress
    
    # Lévy flight component
    levy = np.random.randn(*Positions.shape) * (rg/(1+progress))**0.5
    
    # Hybrid velocity update
    velocity = w * levy + \
              c1 * np.random.rand() * (Best_pos - Positions) + \
              c2 * np.random.rand() * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    
    # Smart boundary handling
    new_pos = Positions + velocity * rg
    mask = (new_pos < lb_array) | (new_pos > ub_array)
    Positions = np.where(mask, 
                        lb_array + (ub_array-lb_array)*np.random.rand(*Positions.shape),
                        new_pos)
    #EVOLVE-END       
    
    return Positions