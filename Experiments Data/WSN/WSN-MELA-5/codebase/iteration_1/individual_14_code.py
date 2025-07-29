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
    w = 0.9 - (0.5 * (1 - rg))  # Dynamic inertia weight
    c1 = 2.0 * np.exp(-rg)     # Cognitive coefficient
    c2 = 2.0 - c1              # Social coefficient
    
    r1 = np.random.rand(*Positions.shape)
    r2 = np.random.rand(*Positions.shape)
    
    velocity = w * Positions + c1*r1*(Best_pos - Positions) + c2*r2*(Best_pos.mean(0) - Positions)
    Positions = np.clip(velocity, lb_array, ub_array)
    #EVOLVE-END       
    
    return Positions