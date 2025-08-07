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
    F = 0.3 + 0.4 * np.cos(rg * np.pi/100)  # Cosine-adaptive scaling
    CR = 0.95 * np.exp(-rg/75)  # Exponential crossover decay
    
    # Enhanced DE with perturbation
    base = Positions[np.random.permutation(SearchAgents_no)]
    donor = base + F*(Best_pos - Positions) + 0.1*np.random.randn(*Positions.shape)
    mask = np.random.rand(*Positions.shape) < CR
    Positions = np.where(mask, np.clip(donor, lb_array, ub_array), Positions)
    #EVOLVE-END       

    return Positions