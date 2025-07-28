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
    # Velocity update with inertia and learning factors
    w = 0.7 - 0.5*(Best_score/10000)  # Adaptive inertia
    c1, c2 = 2.0, 2.0
    velocity = w*np.random.rand(*Positions.shape) + \
               c1*np.random.rand()*(Best_pos - Positions) + \
               c2*np.random.rand()*(Positions[np.random.permutation(SearchAgents_no)] - Positions)
    
    # Adaptive mutation based on progress
    mutation_prob = 0.1 + 0.4*(1 - Best_score/10000)
    mutation_mask = np.random.rand(*Positions.shape) < mutation_prob
    mutated = Positions + 0.5*(ub_array - lb_array)*np.random.randn(*Positions.shape)
    Positions = np.where(mutation_mask, mutated, Positions + velocity)
    #EVOLVE-END       

    return Positions