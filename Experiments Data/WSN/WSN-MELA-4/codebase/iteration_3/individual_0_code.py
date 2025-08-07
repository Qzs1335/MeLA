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
    # Dynamic adaptive parameters with elite guidance
    progress = np.clip(1 - (Best_score/1000), 0.1, 0.9)
    w = 0.9 - 0.5*progress  # Balanced inertia
    c1 = 1.5*(1-progress)    # Cognitive decay
    c2 = 1.5*progress        # Social growth
    
    # Hybrid velocity update
    r1, r2 = np.random.rand(2)
    elite_mask = (np.random.rand(SearchAgents_no,1) < 0.2)
    guidance = elite_mask * Best_pos + (1-elite_mask)*Positions[np.random.permutation(SearchAgents_no)]
    
    velocity = w*np.random.randn(*Positions.shape) + \
              c1*r1*(Best_pos-Positions) + \
              c2*r2*(guidance-Positions)
    
    # Smart boundary handling
    new_pos = Positions + velocity*rg
    out_of_bounds = (new_pos < lb_array) | (new_pos > ub_array)
    Positions = np.where(out_of_bounds, 
                        Positions - 0.5*velocity*rg,  # Partial reflection
                        new_pos)
    #EVOLVE-END       
    
    return Positions