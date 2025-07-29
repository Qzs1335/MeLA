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
    # Adaptive weights
    w = 0.9 - (0.5 * rg)  # Linearly decreasing weight
    
    # Opposition-based learning
    opposite_pos = lb_array + ub_array - Positions
    opp_fitness = np.random.rand(SearchAgents_no, 1) < 0.3
    Positions = np.where(opp_fitness, opposite_pos, Positions)
    
    # Memory mechanism
    memory = 0.1 * np.random.randn(*Positions.shape)
    elite_mask = (np.random.rand(SearchAgents_no, 1) < 0.2)
    Positions = np.where(elite_mask, Best_pos + w*memory, Positions)
    #EVOLVE-END       

    return Positions