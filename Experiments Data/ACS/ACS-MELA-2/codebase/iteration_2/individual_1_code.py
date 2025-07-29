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
    nu_rand = np.random.randn(SearchAgents_no, dim) 
    levy_flight = 0.01*nu_rand*(1-rg) + 0.01*np.random.randn(SearchAgents_no, 1)*Best_pos
    
    learn_prob = np.clip(0.6 - 0.2*np.arange(SearchAgents_no)/SearchAgents_no, 0.3, 0.7)
    mask = np.random.rand(SearchAgents_no,1) < learn_prob.reshape(-1,1)
    
    Positions = np.where(mask,
                        Best_pos + 0.5*(1-rg)*(Positions-Best_pos) + levy_flight,
                        Positions + 0.1*np.random.randn(*Positions.shape))
     #EVOLVE-END       
    return Positions