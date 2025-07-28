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
    # Enhanced dynamic search with cosine modulation
    cos_factor = np.cos(rg*np.pi/2)
    perturbation = cos_factor * (Best_pos - Positions) * (0.4 + 0.6*rg)*np.random.randn(*Positions.shape)
    Positions += perturbation
    
    # Adaptive elite guidance
    elite_mask = np.random.rand(SearchAgents_no, dim) < (0.15 + 0.05*rg)
    Positions = np.where(elite_mask, 
                        Best_pos*(1 + 0.5*rg**2*np.random.randn(*Positions.shape)), 
                        Positions)
    
    # Smart boundary handling
    overshoot = Positions > ub_array
    undershoot = Positions < lb_array
    Positions = np.where(overshoot|undershoot, 
                        Best_pos + (2*rg-1)*np.random.rand(*Positions.shape)*(ub_array-lb_array), 
                        Positions)
    #EVOLVE-END       
    
    return Positions