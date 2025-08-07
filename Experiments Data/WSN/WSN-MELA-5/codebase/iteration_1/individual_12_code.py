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
    opposite_pos = lb_array + ub_array - Positions
    select_mask = np.random.rand(SearchAgents_no, dim) < 0.5
    Positions = np.where(select_mask, opposite_pos, Positions)
    
    # Dynamic exploration-exploitation
    w = 0.9 * (1 - np.exp(-5 * rg))
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    cognitive = r1 * (Best_pos - Positions)
    social = r2 * (Best_pos[np.random.randint(0, SearchAgents_no)] - Positions)
    Positions = w * Positions + cognitive + social
    #EVOLVE-END       

    return Positions