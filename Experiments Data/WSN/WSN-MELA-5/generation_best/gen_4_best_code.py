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
    # Adaptive parameters
    w = 0.9 - (0.9-0.4)*rg
    c1 = 2.0 * rg  # Cognitive coefficient
    c2 = 2.0 * (1-rg)  # Social coefficient
    
    # Vectorized neighborhood search
    neighbor_size = max(3, int(SearchAgents_no*0.2))
    neighbors = np.random.randint(0, SearchAgents_no, (SearchAgents_no, neighbor_size))
    distances = np.linalg.norm(Positions[neighbors] - Best_pos, axis=2)
    local_bests = Positions[neighbors[np.arange(SearchAgents_no), np.argmin(distances, axis=1)]]
    
    # Hybrid update with adaptive coefficients
    r1, r2 = np.random.rand(2, SearchAgents_no, 1)
    Positions = w*Positions + c1*r1*(Best_pos-Positions) + c2*r2*(local_bests-Positions)
    #EVOLVE-END       

    return Positions