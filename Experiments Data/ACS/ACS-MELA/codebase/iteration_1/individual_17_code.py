import numpy as np
import numpy as np
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]

    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))

    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(*Positions.shape)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)     #EVOLVE-START
    # Chaotic exploration using Gaussian map
    chaotic_map = np.exp(-5.5 * np.random.rand(*Positions.shape)**2)
    
    # Adaptive weights
    w_max, w_min = 0.9, 0.2
    w = w_max - (w_max-w_min) * (np.arange(SearchAgents_no)/SearchAgents_no)
    
    # Opposition-based components
    opposite_pos = 1 - Positions[np.random.permutation(SearchAgents_no)] 
    r1, r2, r3 = np.random.rand(3)
    
    # Position update
    Positions = (r1 * w.reshape(-1,1) * Best_pos) + (r2 * (1-w).reshape(-1,1) * opposite_pos) + (r3 * chaotic_map)     #EVOLVE-END
    return Positions