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
    levy_step = 0.01*np.random.randn(SearchAgents_no,1)*sigma/(np.abs(np.random.randn(SearchAgents_no,1)+1e-8)**(1/beta))
    
    ranks = np.argsort(np.linalg.norm(Positions-Best_pos,axis=1))
    learn_prob = 0.3 + 0.6*(ranks/SearchAgents_no)
    noise = np.random.randn(*Positions.shape)/np.sqrt(dim)
    Positions = np.where(np.random.rand(*Positions.shape) < learn_prob.reshape(-1,1),
                        (Best_pos + levy_step * (Positions - Positions.mean(axis=0)))*(1+0.1*noise),
                        Positions*(1 + rg*(np.random.rand(*Positions.shape)-0.5)))
    #EVOLVE-END       
    return Positions