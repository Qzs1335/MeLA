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
    # Enhanced dynamic search with non-linear scaling
    cos_rg = np.cos(rg*np.pi/2)
    perturbation = cos_rg * (Best_pos - Positions) * (0.5 + 0.5*rg) * np.random.randn(*Positions.shape)
    Positions += perturbation
    
    # Adaptive elite guidance
    elite_mask = np.random.rand(SearchAgents_no, dim) < rg**2
    Positions = np.where(elite_mask, 
                        Best_pos + (1-rg)*np.random.randn(*Positions.shape), 
                        Positions)
    
    # Reflective boundary handling
    out_of_bounds = (Positions < lb_array) | (Positions > ub_array)
    Positions = np.where(out_of_bounds, 
                        np.clip(2*Best_pos - Positions, lb_array, ub_array) + 0.1*rg*np.random.randn(*Positions.shape),
                        Positions)
    #EVOLVE-END       
    
    return Positions