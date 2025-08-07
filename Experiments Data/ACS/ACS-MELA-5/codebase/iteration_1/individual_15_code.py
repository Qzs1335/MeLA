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
    opposite_pos = 1 - Positions
    combined_pos = np.vstack([Positions, opposite_pos])
    
    # Adaptive parameters
    t = 1 - (1/(1 + np.exp(-Best_score/1000)))
    r1 = np.random.rand() * t
    r2 = np.random.rand() * (1-t)
    
    # Memory mechanism
    memory_pos = 0.5*(Best_pos + Positions.mean(axis=0))
    Positions = r1*combined_pos[:SearchAgents_no] + r2*memory_pos + (1-r1-r2)*np.random.rand(SearchAgents_no, dim)
    #EVOLVE-END       

    return Positions