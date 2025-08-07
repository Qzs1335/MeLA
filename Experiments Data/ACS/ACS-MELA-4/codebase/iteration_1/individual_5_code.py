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
    # Opposition-based learning
    opposition_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(*Positions.shape) < 0.5, opposition_pos, Positions)
    
    # Adaptive mutation
    t = np.exp(-rg)  # Exponential decay
    cosine_term = np.cos(2*np.pi*np.random.rand(*Positions.shape))
    mutation = t * cosine_term * (Best_pos - Positions)
    
    # Hybrid update
    explore_mask = np.random.rand(*Positions.shape) < 0.3
    Positions = np.where(explore_mask,
                        Positions + mutation,
                        (1-t)*Positions + t*Best_pos)
    #EVOLVE-END       
    return Positions