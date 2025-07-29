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
    # Dynamic weight adjustment
    w = 0.9 - (0.5 * (rg/max(1e-9, rg+1)))
    
    # Opposition-based learning
    opposite_pos = lb_array + ub_array - Positions
    if np.random.rand() < 0.3:
        Positions = np.where(np.random.rand(*Positions.shape) < 0.5, opposite_pos, Positions)
    
    # Neighborhood search
    neighborhood = Positions + w * (Best_pos - Positions) * np.random.randn(*Positions.shape)
    Positions = np.where(np.random.rand(*Positions.shape) < 0.7, neighborhood, Positions)
    #EVOLVE-END       
    
    return Positions