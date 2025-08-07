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
    inertia = 0.7 * np.exp(-rg/30)
    cognitive = 1.5 * np.random.rand(SearchAgents_no, dim)
    social = 1.5 * np.random.rand(SearchAgents_no, dim)
    
    velocity = (inertia * Positions + cognitive * (Best_pos - Positions) +
               social * (Best_pos[np.random.randint(0, SearchAgents_no)] - Positions))
    Positions = np.clip(Positions + velocity, 0, 1)
    #EVOLVE-END       
    return Positions