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
    w = 0.9 - (0.5 * rg)  # Adaptive inertia weight
    c1 = 1.5 * np.random.rand()
    c2 = 1.5 * np.random.rand()
    
    # Memory component
    Personal_best = Positions + 0.1 * np.random.randn(*Positions.shape)
    
    # Hybrid update rule
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    cognitive = c1 * r1 * (Personal_best - Positions)
    social = c2 * r2 * (Best_pos - Positions)
    
    # Neighborhood search
    indices = np.random.permutation(SearchAgents_no)
    neighbor_influence = 0.1 * (Positions[indices] - Positions)
    
    Positions = w * Positions + cognitive + social + neighbor_influence
    #EVOLVE-END       

    return Positions