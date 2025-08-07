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
    u = np.random.normal(0,1,(SearchAgents_no,dim))
    v = np.random.normal(0,1,(SearchAgents_no,dim))
    levy_step = 0.001 * u / (np.abs(v)**(1/beta))
    
    learn_prob = 0.5 + 0.5*np.sin(np.pi*(Best_score - np.linalg.norm(Positions-Best_pos,axis=1))/Best_score)
    mask = np.random.rand(SearchAgents_no,dim) < learn_prob.reshape(-1,1)
    Positions = np.where(mask,
                        Best_pos * (1 + levy_step) + Positions * 0.1,
                        Positions + (Best_pos.mean(axis=0) - Positions) * 0.5 * np.random.rand())
    #EVOLVE-END       
    return Positions