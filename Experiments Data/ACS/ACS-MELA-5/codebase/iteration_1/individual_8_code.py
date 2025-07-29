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
    fitness = np.array([np.sum(pos) for pos in combined_pos])  # Simplified fitness for demo
    best_idx = np.argmin(fitness)
    Positions = combined_pos[best_idx:best_idx+SearchAgents_no]
    
    # Adaptive weight
    w = 0.9 - (0.5 * (rg / 100))  # Linearly decreasing from 0.9 to 0.4
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    Positions = w * Positions + r1 * (Best_pos - Positions) + r2 * (Positions.mean(axis=0) - Positions)
    #EVOLVE-END
    
    return Positions