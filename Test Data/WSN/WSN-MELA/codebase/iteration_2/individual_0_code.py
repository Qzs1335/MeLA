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
    # Adaptive perturbation with memory
    adaptive_scale = 0.3 + 0.7 * rg
    perturbation = adaptive_scale * (Best_pos - Positions) * np.random.randn(*Positions.shape)
    Positions += perturbation
    
    # Dynamic elite guidance
    elite_prob = 0.1 + 0.15 * (1 - rg)
    elite_mask = np.random.rand(SearchAgents_no, dim) < elite_prob
    Positions = np.where(elite_mask, 
                        Best_pos + adaptive_scale*np.random.randn(*Positions.shape), 
                        Positions)
    
    # Smart boundary handling
    boundary_violation = (Positions < lb_array) | (Positions > ub_array)
    Positions = np.where(boundary_violation, 
                        Best_pos * (0.5 + 0.5*rg) * np.random.rand(*Positions.shape), 
                        Positions)
    #EVOLVE-END       
    
    return Positions