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
    fitness_weights = 0.5 + (Best_score / (Positions.shape[0] * dim)) * np.random.rand(SearchAgents_no, 1)
    opposition_pos = (lb_array + ub_array) - Positions
    mutation = np.random.normal(0, rg/2, Positions.shape)
    
    # Adaptive selection between exploration and exploitation
    mask = np.random.rand(SearchAgents_no, 1) < (0.3 + 0.5*(1 - rg))
    Positions = np.where(mask,
                       (Positions + opposition_pos)/2 + mutation * (1 - fitness_weights),
                       Best_pos + rg * (Positions - Best_pos) * fitness_weights)
    #EVOLVE-END       
    
    return Positions