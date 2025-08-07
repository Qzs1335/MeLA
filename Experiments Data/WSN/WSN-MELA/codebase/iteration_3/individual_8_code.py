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
    # Hybrid PSO-cosine update
    cos_factor = np.cos(rg*np.pi/2)
    velocity = 0.7*(Positions - Best_pos) + 1.5*rg*np.random.rand(*Positions.shape)*(Best_pos - Positions)
    Positions = Best_pos + cos_factor*velocity
    
    # Dynamic elite sampling
    elite_mask = np.random.rand(SearchAgents_no) < 0.2*rg
    Positions[elite_mask] = Best_pos + rg*np.random.randn(sum(elite_mask), dim)
    
    # Reflective boundary handling
    over_ub = Positions > ub_array
    over_lb = Positions < lb_array
    Positions = np.where(over_ub, 2*ub_array-Positions, np.where(over_lb, 2*lb_array-Positions, Positions))
    #EVOLVE-END       
    
    return Positions