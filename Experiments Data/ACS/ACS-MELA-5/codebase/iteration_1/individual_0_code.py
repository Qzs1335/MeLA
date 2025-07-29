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
    F = 0.5 * (1 + np.sin(rg * np.pi / 2))  # Adaptive scaling
    CR = 0.9 - 0.5 * (rg / 100)  # Dynamic crossover
    
    # Differential evolution with memory
    donor = Positions + F * (Best_pos - Positions + Positions[np.random.permutation(SearchAgents_no)] - Positions)
    mask = np.random.rand(*Positions.shape) < CR
    Positions = np.where(mask, donor, Positions)
    #EVOLVE-END       

    return Positions