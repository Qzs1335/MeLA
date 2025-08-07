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
    F = 0.3 + 0.4 * np.cos(rg * np.pi/100)  # Cosine-based adaptive scaling
    CR = 0.85 - 0.3 * (rg/100)  # Dynamic crossover
    
    # Hybrid differential evolution
    donor1 = Positions + F * (Best_pos - Positions)
    donor2 = Positions + F * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    donor = 0.5*(donor1 + donor2)  # Hybridization
    
    mask = np.random.rand(*Positions.shape) < CR
    Positions = np.where(mask, donor, Positions)
    #EVOLVE-END       

    return Positions