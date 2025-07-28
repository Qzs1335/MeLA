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
    # Enhanced adaptive parameters
    progress = np.clip(1 - (Best_score / 1000), 0.1, 0.9)
    w = 0.5 * (1 + np.cos(np.pi * progress))  # Smoother inertia decay
    c1 = 1.5 * (1 - progress**0.5)           # Non-linear cognitive decay
    c2 = 1.5 + 1.5 * progress                # Progressive social emphasis
    
    # Hybrid velocity update
    r1, r2 = np.random.rand(2)
    cognitive = c1 * r1 * (Best_pos - Positions)
    social = c2 * r2 * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    perturb = 0.1 * np.random.randn(*Positions.shape) * (1 - progress)
    velocity = w * (velocity if 'velocity' in locals() else np.random.randn(*Positions.shape)) + cognitive + social + perturb
    
    # Robust position update
    Positions = np.clip(Positions + velocity * (0.1 + 0.9*rg), lb_array, ub_array)
    #EVOLVE-END       
    return Positions