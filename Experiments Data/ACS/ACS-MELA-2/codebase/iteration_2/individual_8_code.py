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
    beta = 1.5 + 0.5*np.random.rand()
    levy = np.random.randn(SearchAgents_no,dim) * np.random.rand(SearchAgents_no,dim)**(-1/beta)
    
    learn_prob = 0.5 + (Best_score - np.min(np.linalg.norm(Positions-Best_pos,axis=1)))/Best_score
    step_size = rg * (1 - learn_prob).reshape(-1,1)
    Positions = np.where(np.random.rand(*Positions.shape) < learn_prob.reshape(-1,1),
                        Best_pos + step_size*levy,
                        Positions + step_size*(Positions - Positions.mean(axis=0)))
    #EVOLVE-END       
    return Positions