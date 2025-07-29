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
    F = 0.4 + 0.6 * np.sin(np.pi * np.random.rand(SearchAgents_no,1)/2)  # Adaptive scaling factor
    mutate_mask = np.random.rand(*Positions.shape) < (1/dim + 0.1*np.log1p(Best_score))  # Mutation probability
    
    # Differential mutation
    random1 = Positions[np.random.randint(0, SearchAgents_no, SearchAgents_no)]
    random2 = Positions[np.random.randint(0, SearchAgents_no, SearchAgents_no)]
    mutant = Best_pos + F * (random1 - random2)
    
    # Controlled mutation
    Positions = np.where(mutate_mask, 
                        np.clip(mutant, lb_array, ub_array),
                        Positions)
    #EVOLVE-END       
    
    return Positions