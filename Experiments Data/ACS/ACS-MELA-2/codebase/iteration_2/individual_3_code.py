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
    beta = 1.5
    u = np.random.randn(SearchAgents_no,dim)*0.7071
    v = np.random.randn(SearchAgents_no,dim)
    levy_step = 0.01*u/(np.abs(v)**(1/beta))
    
    influence = np.maximum(0.3, 0.7 * (1 - np.linalg.norm(Positions-Best_pos,axis=1)/dim))
    mask = np.random.rand(SearchAgents_no,dim) < influence.reshape(-1,1)
    Positions = np.where(mask, 
                        Best_pos * (1+0.2*levy_step) + 0.3*levy_step*Positions,
                        Positions*(1 + 0.1*(rg*np.random.rand(*Positions.shape)-rg/2)))
    #EVOLVE-END       
    return Positions