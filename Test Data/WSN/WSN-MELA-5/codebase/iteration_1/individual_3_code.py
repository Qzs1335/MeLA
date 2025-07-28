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
    memory = 0.9 * Positions + 0.1 * Best_pos
    adaptive_rg = rg * (1 - np.exp(-Best_score/1000))
    spiral = np.exp(adaptive_rg * np.random.randn(*Positions.shape)) * np.cos(2*np.pi*np.random.rand(*Positions.shape))
    Positions = np.where(np.random.rand(SearchAgents_no,1) < 0.5, 
                        Best_pos + adaptive_rg * spiral,
                        memory + adaptive_rg * np.random.randn(*Positions.shape))
    #EVOLVE-END       
    return Positions