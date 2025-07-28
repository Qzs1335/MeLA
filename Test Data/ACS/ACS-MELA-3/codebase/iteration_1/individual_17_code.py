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
    momentum = 0.3 * (Positions - 0.5)  # Historical movement bias
    progress_factor = 1 / (1 + np.exp(0.01*(Best_score-5000)))  # Dynamic adjustment
    
    # Hybrid guided-random exploration
    exp_rand = progress_factor * np.random.exponential(scale=0.5, size=Positions.shape)
    target_blend = 0.4 * Best_pos + 0.6 * Positions
    
    Positions = np.clip(target_blend + momentum + exp_rand * rg, 0, 1)
    #EVOLVE-END
    
    return Positions