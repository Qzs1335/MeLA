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
    # Refined perturbation with adaptive scaling
    perturbation = 0.3 * rg * (Best_pos - Positions) * (0.5 + np.random.rand(*Positions.shape))
    Positions += perturbation
    
    # Selective elite guidance
    elite_mask = np.random.rand(SearchAgents_no, dim) < np.sqrt(rg)
    Positions = np.where(elite_mask, 
                        Best_pos * (0.9 + 0.2*rg*np.random.rand(*Positions.shape)),
                        Positions)
    
    # Improved boundary handling with reflection
    boundary_violation = (Positions < lb_array) | (Positions > ub_array)
    Positions = np.where(boundary_violation, 
                        np.clip(2*Best_pos - Positions, lb_array, ub_array),
                        Positions)
    #EVOLVE-END       
    
    return Positions