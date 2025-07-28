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
    w = 0.9 - 0.5*(rg/100)  # Adaptive inertia weight
    cos_factor = 1 - np.abs(np.sum(Positions*Best_pos, axis=1)/(
        np.linalg.norm(Positions,axis=1)*np.linalg.norm(Best_pos)+1e-8))
    
    neighbor_idx = np.random.randint(0, SearchAgents_no, SearchAgents_no)
    neighbor_impulse = Positions[neighbor_idx] * cos_factor.reshape(-1,1)
    
    r1, r2 = np.random.rand(2, SearchAgents_no, 1)
    Positions = w*Positions + r1*(Best_pos - Positions) + r2*(neighbor_impulse - Positions).clip(0,1)
    #EVOLVE-END       

    return Positions