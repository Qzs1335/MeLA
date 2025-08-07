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
    # Adaptive chaotic search
    chaos = 4*np.random.rand(SearchAgents_no, dim)*(1-np.random.rand(SearchAgents_no, dim))
    elite_guidance = Best_pos * (1 + np.sin(np.pi*(np.random.rand()-0.5)))
    
    # Opposition-based learning
    opposite_pos = lb_array + ub_array - Positions
    opposition_mask = np.random.rand(SearchAgents_no, dim) < 0.3
    Positions = np.where(opposition_mask, opposite_pos, Positions)
    
    # Hybrid update
    search_mask = np.random.rand(SearchAgents_no, dim) < 0.5
    Positions = np.where(search_mask,
                        Positions + chaos*(elite_guidance - Positions),
                        Positions + chaos*(Best_pos - Positions))
    #EVOLVE-END       

    return Positions