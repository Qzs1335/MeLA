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
    a = 2 - rg * (2/30)  # Nonlinear decay
    levy = np.random.randn(SearchAgents_no,dim) * 0.01
    momentum = 0.9 * (Best_pos - Positions)
    explo_mask = np.random.rand(SearchAgents_no,dim) > rg
    
    Positions = np.where(explo_mask,
                        a*levy + momentum,
                        Best_pos + np.random.normal(0,1,(SearchAgents_no,dim))*(Best_score/1000))
    #EVOLVE-END       
    return Positions