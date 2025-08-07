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
    stagnation_threshold = 0.1 * Best_score
    mutate_mask = np.random.rand(*Positions.shape) < (0.2 + 0.3*(1-rg))
    
    # Opposition-based learning
    opposition = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(*Positions.shape) < 0.3, opposition, Positions)
    
    # Elite-guided search (fixed missing parenthesis)
    elite_guide = Best_pos * (1 + 0.5*(np.random.rand(*Positions.shape)-0.25))
    Positions = np.where(mutate_mask, elite_guide, Positions)
    
    # Adaptive local search
    local_step = 0.1 * rg * (np.random.rand(*Positions.shape)-0.5)
    Positions = np.clip(Positions + local_step, lb_array, ub_array)
    #EVOLVE-END       

    return Positions