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
    # Adaptive parameter control
    w = 0.9 - (0.5 * rg)  
    c1 = 1.5 * rg
    c2 = 1.5 - c1
    
    # Opposition-based learning
    opp_pos = 1 - Positions
    opp_fit = np.sum(opp_pos, axis=1)
    mask = opp_fit < Best_score
    Positions[mask] = opp_pos[mask]
    
    # Memory-based update
    memory = w * Positions + c1 * np.random.rand() * (Best_pos - Positions) + c2 * np.random.rand() * (Positions.mean(axis=0) - Positions)
    #EVOLVE-END       

    return Positions