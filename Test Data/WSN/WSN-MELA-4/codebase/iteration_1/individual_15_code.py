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
    # Adaptive exploration rate based on iteration progress
    exploration_prob = 0.5 * (1 + np.cos(np.pi * rg))
    
    # Memory-based perturbation
    memory_effect = 0.2 * np.random.randn(*Positions.shape) * (Best_pos - Positions)
    
    # Diversity preservation
    center = np.mean(Positions, axis=0)
    diversity = Positions - center
    
    # Combined update
    r = np.random.rand(SearchAgents_no, 1)
    mask = r < exploration_prob
    Positions = np.where(mask, 
                        Positions + rg * diversity * (0.5 - np.random.rand(*Positions.shape)),
                        Best_pos + memory_effect + 0.1 * np.random.randn(*Positions.shape))
    #EVOLVE-END
    
    return Positions