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
    cos_factor = np.cos(rg * np.pi/180)  # Cosine-based scaling
    F = 0.3 + 0.4 * cos_factor  # Adaptive scaling factor
    CR = 0.85 - 0.3*(1 - cos_factor)  # Dynamic crossover rate
    
    # Hybrid mutation with elite guidance
    elite_term = F * (Best_pos - Positions)
    rand_term = Positions[np.random.permutation(SearchAgents_no)] - Positions[np.random.permutation(SearchAgents_no)]
    donor = Positions + elite_term + F * rand_term
    
    mask = np.random.rand(*Positions.shape) < CR
    Positions = np.where(mask, donor, Positions)
    #EVOLVE-END       

    return Positions