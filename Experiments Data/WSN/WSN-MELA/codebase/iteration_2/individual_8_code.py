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
    # Smoother dynamic search
    perturbation = 0.2 * (1 - rg) * (Best_pos - Positions) * np.random.randn(*Positions.shape)
    Positions += perturbation
    
    # Adaptive elite guidance
    elite_prob = 0.1 + 0.1*rg  # Range 0.1-0.2
    elite_mask = np.random.rand(SearchAgents_no, dim) < elite_prob
    Positions = np.where(elite_mask, 
                        Positions + 0.5*rg*(Best_pos - Positions) + 0.1*np.random.randn(*Positions.shape),
                        Positions)
    
    # Conservative boundary repair
    boundary_violation = (Positions < lb_array) | (Positions > ub_array)
    Positions = np.where(boundary_violation, 
                        0.5*(Positions + Best_pos) + 0.05*rg*np.random.randn(*Positions.shape),
                        Positions)
    #EVOLVE-END       
    
    return Positions