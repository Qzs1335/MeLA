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
    adaptive_weights = np.exp(-np.arange(SearchAgents_no)/10).reshape(-1,1)
    memory_effect = 0.7*Positions + 0.3*np.random.permutation(Positions)
    
    gradient = Best_pos - Positions
    dynamic_step = rg*(1 + np.random.randn(*Positions.shape)*0.2)
    neighborhood = Positions + dynamic_step*np.sign(gradient)*adaptive_weights
    
    Positions = np.where(np.random.rand(*Positions.shape)<0.5, memory_effect, neighborhood)
    #EVOLVE-END       
    return Positions