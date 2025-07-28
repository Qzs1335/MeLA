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
    beta = 1.5 - 0.2*np.random.rand()
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*(2**((beta-1)/2))))**(1/beta)
    levy_step = 0.01*rg * np.random.randn(SearchAgents_no,1) * sigma / (np.abs(np.random.randn(SearchAgents_no,1)+1e-8)**(beta))
    
    learn_prob = 0.5 + 0.4*(Positions @ Best_pos)/(np.linalg.norm(Positions,axis=1)*np.linalg.norm(Best_pos)+1e-8)
    mask = (np.random.rand(SearchAgents_no,dim) < learn_prob.reshape(-1,1)) & (np.random.rand(SearchAgents_no,dim)>0.2)
    Positions = np.where(mask,
                        (1-learn_prob).reshape(-1,1)*Best_pos + levy_step*(Positions - Positions[0]),
                        Positions*(0.9 + 0.2*np.random.rand())) 
    #EVOLVE-END       
    return Positions