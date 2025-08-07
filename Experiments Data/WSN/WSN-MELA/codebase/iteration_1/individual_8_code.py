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
    # Dynamic exploration-exploitation balance
    progress = 1 - (rg / 2.28)  # Normalized based on initial rg=2.28
    w = 0.9 * progress + 0.1  # Decreasing inertia weight
    
    # Memory-guided perturbation
    memory_pos = np.mean(Positions[:SearchAgents_no//3], axis=0)
    perturbation = 0.5 * (1-progress) * (memory_pos - Positions)
    
    # Adaptive neighborhood search
    neighbor_mask = np.random.rand(SearchAgents_no, dim) < progress
    neighbor_influence = np.random.rand(*Positions.shape) * (Best_pos - Positions)
    
    Positions = w * Positions + perturbation + neighbor_mask * neighbor_influence
    #EVOLVE-END
    
    return Positions