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
    decay = np.exp(-0.1*rg)
    mutation_prob = 0.2 * decay
    
    scale = (1 + Best_score)/Best_score if Best_score != 0 else 1
    perturb_mask = np.random.rand(*Positions.shape) < mutation_prob
    
    mutant = Best_pos + scale * (np.random.randn(SearchAgents_no, dim) * 1/rg)
    Positions = np.where(perturb_mask, 
                        mutant,
                        Positions * (0.9 + 0.2*np.random.rand()))
    #EVOLVE-END       
    
    return Positions