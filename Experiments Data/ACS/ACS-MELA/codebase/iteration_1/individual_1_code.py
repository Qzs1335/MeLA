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
    # Balance exploration and exploitation with nonlinear weight
    w = 0.9 * (1 - (rg/10)**2)
    r1, r2 = np.random.rand(2)
    
    # Ensure dimension compatibility
    cognitive = 1.5 * r1 * (Best_pos.reshape(1, -1) - Positions)
    social = 1.5 * r2 * (Positions[np.random.randint(SearchAgents_no)] - Positions)
    
    # Random permutation for diversity with proper dimension handling
    if rg % 3 == 0:
        permutation = np.random.permutation(Positions.shape[0])
        Positions = Positions[permutation]

    # Update with proper velocity calculation
    velocities = w * Positions + cognitive + social
    Positions = np.clip(velocities, 0, 1)
    #EVOLVE-END       

    return Positions