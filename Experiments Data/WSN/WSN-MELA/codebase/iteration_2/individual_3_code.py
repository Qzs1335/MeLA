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
    # Enhanced dynamic search with cosine adaptation
    cos_factor = np.cos(rg*np.pi/2)
    perturbation = cos_factor * (Best_pos - Positions) * (0.5 + 0.5*rg) * np.random.randn(*Positions.shape)
    Positions += perturbation
    
    # Memory-based elite guidance
    elite_mask = np.random.rand(SearchAgents_no, dim) < (0.1 + 0.1*rg)
    elite_guide = Best_pos + (1-rg)*np.random.randn(*Positions.shape)
    Positions = np.where(elite_mask, elite_guide, Positions)
    
    # Reflective boundary handling
    over_ub = Positions > ub_array
    under_lb = Positions < lb_array
    Positions = np.where(over_ub, 2*ub_array - Positions, Positions)
    Positions = np.where(under_lb, 2*lb_array - Positions, Positions)
    #EVOLVE-END       
    
    return Positions