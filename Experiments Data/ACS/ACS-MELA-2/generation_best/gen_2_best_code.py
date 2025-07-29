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
    # Levy flight simplified
    u = np.random.randn(SearchAgents_no, dim)*0.001
    v = np.random.randn(SearchAgents_no, dim)
    levy = u/np.power(np.abs(v), 1/1.5)
    
    # Focused learning with adaptive step
    alpha = 0.3*(Best_score/np.linalg.norm(Positions-Best_pos, axis=1).max())
    r_mask = np.random.rand(SearchAgents_no, dim) < 0.7
    Positions = np.where(r_mask, 
                        Best_pos + alpha*levy*(Positions-Best_pos),
                        Best_pos*(1 + 0.4*np.random.randn(SearchAgents_no, dim)))
    #EVOLVE-END       
    return Positions