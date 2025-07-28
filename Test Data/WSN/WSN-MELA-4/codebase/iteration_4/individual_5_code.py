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
    # Hybrid adaptive parameters
    progress = rg/2.28  # Use rg for dynamic scaling (2.28 is initial rg)
    w = 0.5 * (1 + progress)
    c1 = 1.5 * (1 - progress)
    c2 = 1.5 - c1
    
    # Velocity-cosine hybrid update
    velocity = w * np.random.randn(*Positions.shape) + \
               c1 * np.random.rand() * (Best_pos - Positions) + \
               c2 * np.random.rand() * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    
    # Cosine-based position perturbation
    theta = 2*np.pi*np.random.rand(SearchAgents_no,1)
    cos_factor = np.cos(theta) * (1-progress)
    Positions = Positions + velocity*rg + cos_factor*(Best_pos-Positions)*rg
    
    # Robust boundary handling
    overflow = Positions > ub_array
    underflow = Positions < lb_array
    Positions = np.where(overflow | underflow, 
                        np.clip(Positions, lb_array, ub_array) + 
                        np.random.rand(*Positions.shape)*(ub_array-lb_array)*0.1,
                        Positions)
    #EVOLVE-END       
    return Positions