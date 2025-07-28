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
    # Adaptive momentum and diversity
    momentum = 0.9 - 0.5 * (rg / 10)
    rand_mask = np.random.rand(*Positions.shape) < 0.1
    
    # Hybrid updates
    cognitive = np.random.rand(SearchAgents_no, dim) * (Best_pos - Positions)
    social = np.random.rand(SearchAgents_no, dim) * (Positions.mean(axis=0) - Positions)
    
    Positions = (momentum * Positions + 
                0.5 * cognitive + 
                0.3 * social + 
                0.2 * rand_mask * np.random.randn(*Positions.shape))
    #EVOLVE-END       

    return Positions