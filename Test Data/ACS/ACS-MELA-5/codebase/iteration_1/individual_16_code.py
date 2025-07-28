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
    w = 0.9 - 0.5 * (1 - 1/(1 + np.exp(-Best_score/1000)))  # Adaptive inertia weight
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Opposition-based learning
    opposite_pos = lb_array + ub_array - Positions
    opposite_pos = np.where((opposite_pos < lb_array) | (opposite_pos > ub_array), rand_adjust, opposite_pos)
    
    # Elite guidance with cosine modulation
    theta = np.random.rand() * 2 * np.pi
    elite_guide = Best_pos + np.cos(theta) * (Best_pos - Positions.mean(axis=0))
    
    # Combined update
    Positions = w * Positions + (1-w) * (r1 * opposite_pos + r2 * elite_guide)
    #EVOLVE-END
    
    return Positions