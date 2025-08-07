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
    F = 0.3 + 0.4 * (1 + np.cos(rg * np.pi/100))  # Cosine-modulated scaling
    CR = 0.85 - 0.45 * (rg % 2)  # Oscillating crossover
    
    # Hybrid mutation with dimensional checks
    perm_idx = np.random.permutation(SearchAgents_no)
    donor = Positions + F * (Best_pos - Positions + Positions[perm_idx] - Positions[np.random.permutation(SearchAgents_no)])
    mask = (np.random.rand(*Positions.shape) < CR) & (np.arange(dim) % 3 != 0)  # Dimensional mask
    Positions = np.where(mask, donor, Positions)
    #EVOLVE-END       
    return Positions