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
    F = 0.3 + 0.4 * np.cos(rg * np.pi/100)  # Cosine-modulated scaling
    CR = 0.85 - 0.3 * (rg / 100)  # Dynamic crossover
    
    # Hybrid mutation: best-guided + random permutation
    elite_mask = np.random.rand(SearchAgents_no) < 0.7
    donor_elite = Best_pos + F * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    donor_rand = Positions[np.random.permutation(SearchAgents_no)] + F * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    donor = np.where(elite_mask[:,None], donor_elite, donor_rand)
    
    mask = np.random.rand(*Positions.shape) < CR
    Positions = np.where(mask, donor, Positions)
    #EVOLVE-END       

    return Positions