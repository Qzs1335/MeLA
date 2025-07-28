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
    w = 0.9 * (1 - np.exp(-0.05 * rg))  # Adaptive weight
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    
    cognitive = 1.5 * r1 * (Best_pos - Positions)
    social_vec = Positions[np.random.permutation(SearchAgents_no)]
    social = 1.5 * r2 * (social_vec - Positions)
    
    noise = 0.05 * (2 * np.random.rand(SearchAgents_no, dim) - 1)
    Positions = Positions + w * Positions + cognitive + social + noise
    #EVOLVE-END
    
    return Positions