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
    F = 0.5 * (1 + np.sin(2 * np.pi * rg / 10))  # Adaptive scaling
    CR = 0.9 * (1 - rg/3)  # Dynamic crossover
    
    # Differential evolution mutation
    idxs = np.random.permutation(SearchAgents_no)[:3]
    mutant = Positions[idxs[0]] + F * (Positions[idxs[1]] - Positions[idxs[2]])
    
    # Hybrid update
    mask = np.random.rand(*Positions.shape) < CR
    Positions = np.where(mask, 
                        mutant + 0.1*rg*(Best_pos - Positions),
                        Positions + rg*(Best_pos - Positions))
    #EVOLVE-END       

    return Positions