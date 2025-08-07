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
    # Opposition-based learning for diversity
    opp_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no,dim) < 0.5, Positions, opp_pos)
    
    # Adaptive weights
    w = 0.9 - 0.5*(rg/2.28)  # Dynamic inertia weight
    c1 = 2.5 - 2*(rg/2.28)    # Cognitive component
    c2 = 0.5 + 2*(rg/2.28)    # Social component
    
    # Hybrid update
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    velocity = w*Positions + c1*r1*(Best_pos - Positions) + c2*r2*(Best_pos.mean(0) - Positions)
    Positions = np.clip(velocity, lb_array, ub_array)
    #EVOLVE-END       

    return Positions