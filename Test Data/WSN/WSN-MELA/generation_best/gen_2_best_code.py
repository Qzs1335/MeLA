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
    # Enhanced dynamic search with momentum
    momentum = 0.9 - 0.8*rg  # Decaying momentum
    perturbation = momentum * (Best_pos - Positions) * np.random.randn(*Positions.shape)
    Positions += 0.7*rg * perturbation
    
    # Adaptive elite guidance
    elite_prob = 0.3*(1-rg)  # Decreasing probability
    elite_mask = np.random.rand(SearchAgents_no, dim) < elite_prob
    Positions = np.where(elite_mask, 
                        Best_pos + 0.5*rg*np.random.randn(*Positions.shape), 
                        Positions)
    
    # Smart boundary handling
    boundary_violation = (Positions < lb_array) | (Positions > ub_array)
    rebound_factor = 0.2 + 0.3*rg  # Adaptive rebound
    Positions = np.where(boundary_violation, 
                        Best_pos + rebound_factor*np.random.randn(*Positions.shape), 
                        Positions)
    #EVOLVE-END       
    
    return Positions