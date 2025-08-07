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
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(SearchAgents_no, dim) * sigma
    v = np.random.randn(SearchAgents_no, dim)
    levy_step = 0.01 * u / (np.abs(v)**(1/beta))
    
    learn_prob = np.clip(0.5 + 0.4*(Best_score - np.linalg.norm(Positions-Best_pos,axis=1))/Best_score, 0.1, 0.9)
    mask = np.random.rand(SearchAgents_no,dim) < learn_prob.reshape(-1,1)
    Positions = np.where(mask,
                        Best_pos + levy_step*(Positions - Positions.mean(axis=0)),
                        Positions + 0.5*levy_step*(Positions - Positions[np.random.randint(0, SearchAgents_no)]))
    #EVOLVE-END       
    return Positions