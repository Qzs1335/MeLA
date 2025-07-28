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
    F = 0.5 * (1 + np.cos(rg * np.pi / 4))  # Cosine-adaptive scaling
    CR = 1 / (1 + np.exp(-0.1*(rg-50)))  # Sigmoid crossover rate
    
    # Hybrid differential evolution
    historical_best = 0.7*Best_pos + 0.3*Positions.mean(axis=0)
    donor = historical_best + F * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    mask = np.random.rand(*Positions.shape) < CR
    Positions = np.clip(np.where(mask, donor, Positions), lb_array, ub_array)
    #EVOLVE-END       

    return Positions