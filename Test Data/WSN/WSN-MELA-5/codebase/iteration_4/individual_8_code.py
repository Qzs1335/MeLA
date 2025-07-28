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
    w = 0.9 * (1 - rg**2)  # Non-linear decay
    neighbor_size = max(3, int(SearchAgents_no*(0.3 - 0.1*rg)))
    
    # Vectorized neighborhood processing
    neighbors = np.random.randint(0, SearchAgents_no, (SearchAgents_no, neighbor_size))
    neighbor_pos = Positions[neighbors]
    dists = np.linalg.norm(neighbor_pos - Best_pos, axis=2)
    local_bests = neighbor_pos[np.arange(SearchAgents_no), np.argmin(dists, axis=1)]
    
    r1, r2 = np.random.rand(2, SearchAgents_no, 1)
    cognitive = (1.5 - rg) * r1 * (Best_pos - Positions)
    social = (1.0 + rg) * r2 * (local_bests - Positions)
    Positions = w*Positions + cognitive + social
    #EVOLVE-END       

    return Positions