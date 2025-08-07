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
    # Velocity component with inertia
    velocity = 0.7 * np.random.randn(*Positions.shape) + 0.3 * (Best_pos - Positions)
    
    # Adaptive convergence factor
    conv_factor = 1 - np.exp(-3 * (1 - rg))
    
    # Hybrid update with exploration component
    r = np.random.rand(SearchAgents_no, 1)
    mask = r < conv_factor
    Positions = np.where(mask, 
                        Positions + velocity * conv_factor,
                        Best_pos + rg * np.random.randn(*Positions.shape))
    #EVOLVE-END
    
    return Positions