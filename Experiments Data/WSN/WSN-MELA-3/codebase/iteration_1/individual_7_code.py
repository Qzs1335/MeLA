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
    # Dynamic inertia weight
    w = 0.9 - (0.5 * (rg / 2.28))  # Linearly decreases with rg
    
    # Neighborhood search
    neighbor_size = max(1, int(SearchAgents_no * 0.2))
    neighbors = np.random.choice(SearchAgents_no, (SearchAgents_no, neighbor_size), replace=False)
    local_best = Positions[np.arange(SearchAgents_no), np.argmin(Positions[neighbors], axis=1)]
    
    # Hybrid update
    r1, r2 = np.random.rand(2, SearchAgents_no, dim)
    cognitive = 1.5 * r1 * (Best_pos - Positions)
    social = 1.5 * r2 * (local_best - Positions)
    Positions = w * Positions + cognitive + social
    #EVOLVE-END
    
    return Positions