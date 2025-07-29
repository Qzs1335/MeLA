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
    # Adaptive components
    w = 0.7 + 0.2*np.cos(rg*np.pi/2)  # Non-linear inertia
    neighbor_size = max(2, int(SearchAgents_no*(0.1 + 0.1*rg)))
    
    # Dual-phase update
    for i in range(SearchAgents_no):
        # Local-best selection
        dists = np.linalg.norm(Positions - Positions[i], axis=1)
        neighbors = np.argpartition(dists, neighbor_size)[:neighbor_size]
        local_best = Positions[neighbors[np.argmin(dists[neighbors])]]
        
        # Hybrid update with dynamic coefficients
        r1, r2 = np.random.rand(2)
        cognitive = (1.5 - rg) * r1 * (Best_pos - Positions[i])
        social = (1.0 + rg) * r2 * (local_best - Positions[i])
        Positions[i] = w*Positions[i] + cognitive + social
    #EVOLVE-END       

    return Positions