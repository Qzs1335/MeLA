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
    prob_guided = 0.4 + (0.6 * (Best_score - Positions.min())/(Best_score + 1e-8))
    mask1 = np.random.rand(*Positions.shape) < prob_guided
    mask2 = np.random.rand(*Positions.shape) < (1-prob_guided)*0.5
    
    guided_term = Best_pos.reshape(1,-1) * (0.9 + 0.2*rg*np.random.randn(*Positions.shape))
    crossover_term = Positions[np.random.permutation(SearchAgents_no)] * Positions
    
    Positions = np.where(mask1, guided_term, 
                    np.where(mask2, crossover_term,
                    Positions * (0.5 + rg*np.random.rand(*Positions.shape))))
    #EVOLVE-END       
    
    return Positions