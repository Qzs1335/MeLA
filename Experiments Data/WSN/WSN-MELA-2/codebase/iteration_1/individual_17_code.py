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
    combined_pos = np.vstack((Positions, opposite_pos))
    
    # Dynamic weights
    w = 0.9 - (0.5 * (rg / 2.28))  # Linearly decreasing from 0.9 to 0.4
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Adaptive mutation
    mutation_prob = 0.1 * (1 - (rg / 2.28))
    mutation_mask = np.random.rand(*Positions.shape) < mutation_prob
    mutation = 0.1 * (ub_array - lb_array) * np.random.randn(*Positions.shape)
    
    # Update positions
    cognitive = r1 * (Best_pos - Positions)
    social = r2 * (combined_pos[np.random.choice(2*SearchAgents_no, SearchAgents_no)] - Positions)
    Positions = w * Positions + cognitive + social + mutation * mutation_mask
    #EVOLVE-END
    
    return Positions