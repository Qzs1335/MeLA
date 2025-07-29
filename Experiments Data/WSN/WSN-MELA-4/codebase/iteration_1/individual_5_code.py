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
    # Adaptive search parameters
    w = 0.7 * (1 - np.exp(-0.01 * (100 - rg*10)))  # Inertia weight decay
    c1 = 1.5 * np.random.rand()
    c2 = 1.5 * np.random.rand()
    
    # Memory-based perturbation
    memory = Positions[np.random.permutation(SearchAgents_no)]
    perturbation = 0.1 * rg * (memory - Positions)
    
    # Hybrid update
    r1, r2 = np.random.rand(2, SearchAgents_no, dim)
    cognitive = c1 * r1 * (Best_pos - Positions)
    social = c2 * r2 * (memory - Positions)
    Positions = w * Positions + cognitive + social + perturbation
    
    # Random restart for 10% agents
    restart_mask = np.random.rand(SearchAgents_no) < 0.1
    Positions[restart_mask] = np.random.rand(np.sum(restart_mask), dim)
    #EVOLVE-END       

    return Positions