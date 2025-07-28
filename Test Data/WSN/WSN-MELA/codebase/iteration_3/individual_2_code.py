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
    # Cosine-based adaptive scaling
    cos_factor = np.cos(rg*np.pi/2) 
    velocity = cos_factor * (Best_pos - Positions) * np.random.rand(*Positions.shape)
    
    # Elite guidance with memory
    pb_pos = Positions + 0.5*(Best_pos - Positions) 
    elite_prob = 0.3*(1-rg) 
    elite_mask = np.random.rand(*Positions.shape) < elite_prob
    Positions = np.where(elite_mask, pb_pos + rg*np.random.randn(*Positions.shape), Positions + velocity)
    
    # Reflective boundary handling
    out_of_bounds = (Positions < lb_array) | (Positions > ub_array)
    Positions = np.where(out_of_bounds, 0.5*(Best_pos + Positions), Positions)
    #EVOLVE-END       
    
    return Positions