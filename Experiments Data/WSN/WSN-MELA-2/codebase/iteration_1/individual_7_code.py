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
    # Adaptive search parameters
    w = 0.9 - 0.5*(rg/2.28)  # Linearly decreasing inertia
    c1 = 2.0 - (rg/2.28)      # Cognitive coefficient
    c2 = 2.0                  # Social coefficient
    
    # Velocity update with memory
    velocity = w * np.random.rand(*Positions.shape) + \
              c1 * np.random.rand() * (Best_pos - Positions) + \
              c2 * np.random.rand() * (Positions.mean(axis=0) - Positions)
    
    # Local search around best solution
    if rg < 1.0:
        mask = np.random.rand(*Positions.shape) < 0.3
        Positions = np.where(mask, 
                           Best_pos + 0.1*rg*np.random.randn(*Positions.shape),
                           Positions + velocity)
    else:
        Positions += velocity
    #EVOLVE-END       
    
    return Positions