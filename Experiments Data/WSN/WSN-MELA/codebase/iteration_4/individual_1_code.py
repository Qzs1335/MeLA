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
    # Enhanced dynamic search with Levy flights
    levy = np.random.randn(*Positions.shape) * np.abs(np.random.randn(*Positions.shape))**-1.5
    perturbation = 0.5*(1-rg)*(Best_pos - Positions)*levy
    Positions += perturbation
    
    # Adaptive elite guidance
    elite_prob = 0.1 + 0.1*rg  # Dynamic probability scaling
    elite_mask = np.random.rand(*Positions.shape) < elite_prob
    Positions = np.where(elite_mask, 
                        Best_pos + 0.5*rg*levy, 
                        Positions)
    
    # Reflective boundary handling
    over = Positions > ub_array
    under = Positions < lb_array
    Positions = np.where(over, 2*ub_array - Positions, Positions)
    Positions = np.where(under, 2*lb_array - Positions, Positions)
    #EVOLVE-END       
    
    return Positions