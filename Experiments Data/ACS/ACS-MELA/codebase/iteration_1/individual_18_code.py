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
    inertia = 0.9 * (1 - np.min([0.9, Best_score/10000]))  # Adaptive inertia
    cognitive = 0.5 + 0.4 * np.random.rand(SearchAgents_no, dim)
    social = 0.3 + 0.7 * np.random.rand(SearchAgents_no, dim)
    
    velocity = inertia * (np.random.rand(*Positions.shape) - 0.5) + \
              cognitive * (Best_pos - Positions) + \
              social * (np.mean(Positions, axis=0) - Positions)
    
    Positions = Positions + velocity * rg
    mutation_mask = np.random.rand(*Positions.shape) < 0.1
    Positions = np.where(mutation_mask, Positions + (0.2*rg)*(np.random.rand(*Positions.shape)-0.5), Positions)
    #EVOLVE-END       
    
    return Positions