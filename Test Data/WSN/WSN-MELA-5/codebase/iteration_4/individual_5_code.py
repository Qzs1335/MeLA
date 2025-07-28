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
    w = 0.5 * (1 + np.cos(np.pi * rg))  # Cosine-based inertia decay
    
    # Vectorized neighborhood search
    neighbor_size = max(3, int(SearchAgents_no*0.2))
    neighbors = np.random.randint(0, SearchAgents_no, (SearchAgents_no, neighbor_size))
    dists = np.linalg.norm(Positions[neighbors] - Best_pos, axis=2)
    local_bests = Positions[neighbors[np.arange(SearchAgents_no), np.argmin(dists, axis=1)]]
    
    # Non-linear hybrid update
    r = np.random.rand(SearchAgents_no, 2, dim)
    cognitive = 1.7 * r[:,0] * (Best_pos - Positions)
    social = 1.7 * r[:,1] * (local_bests - Positions)
    Positions = w * Positions + (1-w) * (cognitive + social)
    #EVOLVE-END       

    return Positions