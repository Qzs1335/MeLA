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
    F = 0.3 + 0.4 * (1 + np.sin(rg * np.pi / 4))  # Smoother adaptive scaling
    CR = 0.95 - 0.6 * (rg / 100)  # Slower crossover decay
    
    # Enhanced differential evolution
    permuted = Positions[np.random.permutation(SearchAgents_no)]
    donor = Best_pos + F * (permuted - Positions[np.random.permutation(SearchAgents_no)])
    mask = np.random.rand(*Positions.shape) < CR
    Positions = np.where(mask, donor, Positions)
    #EVOLVE-END       

    return Positions