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
    # Adaptive velocity update
    velocity = 0.5 * np.random.randn(*Positions.shape) + 0.5 * (Best_pos - Positions)
    
    # Random restart for 10% agents
    restart_mask = np.random.rand(SearchAgents_no) < 0.1
    Positions[restart_mask] = np.random.rand(np.sum(restart_mask), dim)
    
    # rg-based neighborhood search
    neighborhood = Positions + rg * np.random.randn(*Positions.shape)
    Positions = np.where(np.random.rand(*Positions.shape) < 0.7, 
                        Positions + velocity, 
                        neighborhood)
    #EVOLVE-END       

    return Positions