import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]

    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))

    #EVOLVE-START
    F = 0.3 + 0.7 * np.cos(rg * np.pi/200)  # Cosine-adaptive scaling
    CR = 0.85 * (1 - np.exp(-rg/50))  # Exponential crossover
    
    # Hybrid DE with boundary protection
    donor = Best_pos + F * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    mask = np.random.rand(*Positions.shape) < CR
    Positions = np.where(mask, donor, Positions)
    Positions = np.clip(Positions, lb_array, ub_array)
    #EVOLVE-END       

    return Positions