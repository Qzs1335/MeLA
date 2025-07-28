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
    F = 0.3 + 0.4 * np.cos(rg * np.pi / 200)  # Cosine-modulated scaling
    CR = 0.85 * (1 - rg/150)  # Dynamic decay crossover
    
    # Hybrid mutation with elite guidance
    perm1, perm2 = np.random.permutation(SearchAgents_no), np.random.permutation(SearchAgents_no)
    donor = Positions + F*(Best_pos - Positions) + F*(Positions[perm1] - Positions[perm2])
    
    # Crossover with boundary check
    mask = (np.random.rand(*Positions.shape) < CR) & (np.abs(donor) < 1.5)
    Positions = np.where(mask, np.clip(donor, 0, 1), Positions)
    #EVOLVE-END       

    return Positions