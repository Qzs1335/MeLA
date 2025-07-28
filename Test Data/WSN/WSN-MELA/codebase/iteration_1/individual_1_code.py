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
    # Dynamic neighborhood search
    perturbation = 0.5 * (1 - rg) * (Best_pos - Positions) * np.random.randn(*Positions.shape)
    Positions += perturbation
    
    # Elite-guided exploration
    elite_mask = np.random.rand(SearchAgents_no, dim) < 0.2*rg
    Positions = np.where(elite_mask, 
                        Best_pos + rg*np.random.randn(*Positions.shape), 
                        Positions)
    
    # Adaptive boundary handling
    boundary_violation = (Positions < lb_array) | (Positions > ub_array)
    Positions = np.where(boundary_violation, 
                        Best_pos + 0.1*rg*np.random.randn(*Positions.shape), 
                        Positions)
    #EVOLVE-END       
    
    return Positions