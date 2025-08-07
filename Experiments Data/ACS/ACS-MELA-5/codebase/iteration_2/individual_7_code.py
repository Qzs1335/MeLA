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
    F = 0.3 + 0.4 * (1 - Best_score/(Best_score+rg))  # Score-adaptive scaling
    CR = 0.8 * np.exp(-rg/50)  # Exponential decay crossover
    cos_perturb = np.cos(np.pi*rg/100) * np.random.randn(*Positions.shape)
    
    donor = Positions + F*(Best_pos - Positions) + cos_perturb
    mask = (np.random.rand(*Positions.shape) < CR) | (np.random.rand(*Positions.shape) < 0.1)
    Positions = np.where(mask, donor, Positions)
    #EVOLVE-END       

    return Positions