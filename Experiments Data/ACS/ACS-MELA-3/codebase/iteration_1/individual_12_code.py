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
    mutation_rate = 0.1 + 0.4 * (1 - np.exp(-Best_score/rg))
    omega = 0.3 + 0.7 * np.cos(np.pi * rg / 100)
    
    elite_mask = np.random.rand(*Positions.shape) < 0.5
    Positions = omega * Positions + (1-omega) * (elite_mask*Best_pos + (1-elite_mask)*np.random.rand(*Positions.shape))
    
    restart_mask = np.random.rand(SearchAgents_no,1) < 0.01
    Positions = np.where(restart_mask, rand_adjust, Positions)
    #EVOLVE-END
    
    return Positions