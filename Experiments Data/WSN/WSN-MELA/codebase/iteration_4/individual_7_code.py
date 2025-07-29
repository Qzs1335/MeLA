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
    # Enhanced exploration with cosine-based scaling
    cos_factor = np.cos(rg * np.pi/2) * np.random.rand(SearchAgents_no, 1)
    levy_flight = np.random.randn(*Positions.shape) * rg * (1 - rg)
    
    # Dynamic elite guidance
    elite_thresh = 0.1 + 0.3 * rg
    elite_mask = np.random.rand(SearchAgents_no, dim) < elite_thresh
    Positions = np.where(elite_mask,
                        Best_pos*(1 + rg*cos_factor) + levy_flight,
                        Positions + cos_factor*(Best_pos - Positions))
    
    # Reflective boundary handling
    outside = (Positions < lb_array) | (Positions > ub_array)
    Positions = np.where(outside,
                        np.clip(2*lb_array - Positions, lb_array, ub_array) if np.random.rand() < 0.5 
                        else np.clip(2*ub_array - Positions, lb_array, ub_array),
                        Positions)
    #EVOLVE-END       
    
    return Positions