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
    # Refined dynamic neighborhood search
    perturbation = 0.2 * (1 - rg) * (Best_pos - Positions) * np.random.randn(*Positions.shape)
    Positions += rg * perturbation
    
    # Selective elite guidance
    elite_mask = np.random.rand(SearchAgents_no, dim) < (0.1 + 0.1*rg)
    Positions = np.where(elite_mask, 
                        Best_pos + 0.5*rg*np.random.randn(*Positions.shape), 
                        Positions)
    
    # Smart boundary reflection
    boundary_violation = (Positions < lb_array) | (Positions > ub_array)
    Positions = np.where(boundary_violation, 
                        Best_pos - 0.5*(Positions - Best_pos), 
                        Positions)
    #EVOLVE-END       
    
    return Positions