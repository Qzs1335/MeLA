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
    tetha = 2*np.pi*np.random.rand(SearchAgents_no, dim)
    w = 0.9 - 0.5*(Best_score/rg)
    delta = np.abs(Best_pos - Positions)
    partners = Positions[np.random.permutation(SearchAgents_no)]
    r1, r2 = np.random.rand(2, SearchAgents_no, dim)
    Positions = w*Positions + r1*np.cos(tetha)*delta + r2*(partners - Positions)
    #EVOLVE-END       
    return Positions