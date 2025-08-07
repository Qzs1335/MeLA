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
    fitness = -np.sum(combined_pos * Best_pos.reshape(1,-1), axis=1)  # Simplified fitness
    top_idx = np.argsort(fitness)[:SearchAgents_no]
    Positions = combined_pos[top_idx]
    
    # Non-linear adaptive parameters
    t = rg * (1 - np.exp(-0.1 * Best_score))
    a = 2 * (1 - t)
    r1 = a * (2 * np.random.rand(SearchAgents_no, dim) - 1)
    r2 = 2 * np.random.rand(SearchAgents_no, dim)
    
    # Enhanced search equation
    A = 2 * a * np.random.rand(SearchAgents_no, dim) - a
    C = 2 * np.random.rand(SearchAgents_no, dim)
    D = np.abs(C * Best_pos - Positions)
    Positions = Best_pos - A * D + r1 * np.sin(r2) * (Best_pos - Positions)
    #EVOLVE-END

    return Positions