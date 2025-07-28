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
    # Opposition-based learning
    opposite_pos = lb_array + ub_array - Positions
    select_mask = np.random.rand(SearchAgents_no, dim) < 0.5*rg
    Positions = np.where(select_mask, opposite_pos, Positions)
    
    # Adaptive mutation
    mutation_rate = 0.1 + 0.4*(1 - rg)
    mutation_mask = np.random.rand(SearchAgents_no, dim) < mutation_rate
    mutation = 0.5*(ub_array - lb_array)*np.random.randn(*Positions.shape)
    Positions = np.where(mutation_mask, Positions + mutation, Positions)
    
    # Dynamic weighting
    w = 0.4 + 0.4*np.cos(np.pi*rg/2)
    Positions = w*Positions + (1-w)*Best_pos
    #EVOLVE-END
    
    return Positions