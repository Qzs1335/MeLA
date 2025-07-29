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
    # Dynamic parameters
    w = 0.9 - (0.5*rg)
    neighbor_size = max(2, int(SearchAgents_no*(0.3 - 0.2*rg)))
    
    # Vectorized neighborhood search
    all_neighbors = np.random.randint(0, SearchAgents_no, (SearchAgents_no, neighbor_size))
    neighbor_dists = np.linalg.norm(Positions[all_neighbors] - Best_pos, axis=2)
    local_bests = Positions[all_neighbors[np.arange(SearchAgents_no), np.argmin(neighbor_dists, axis=1)]]
    
    # Hybrid update with turbulence
    r = np.random.rand(SearchAgents_no, dim)
    turbulence = 0.1*rg*(2*r - 1)
    Positions = w*Positions + 1.5*r*(Best_pos - Positions) + 1.5*(1-r)*(local_bests - Positions) + turbulence
    #EVOLVE-END       

    return Positions