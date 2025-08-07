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
    # Adaptive exploration-exploitation factor
    w = 0.9 - (0.5 * rg)
    
    # Neighborhood search with dynamic radius
    neighbor_radius = 0.1 + 0.4 * (1 - rg)
    neighbor_mask = np.random.rand(SearchAgents_no, dim) < neighbor_radius
    
    # Hybrid update combining historical best and current diversity
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    cognitive = r1 * (Best_pos - Positions)
    social = r2 * (Positions.mean(axis=0) - Positions)
    
    Positions = w * Positions + cognitive + social
    Positions = np.where(neighbor_mask, Positions + 0.1*(np.random.rand(*Positions.shape)-0.5), Positions)
    #EVOLVE-END
    
    return Positions