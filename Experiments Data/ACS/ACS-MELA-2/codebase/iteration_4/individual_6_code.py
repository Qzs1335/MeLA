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
    beta = 1.5 - 0.5*(1 - Best_score/np.linalg.norm(Positions).mean())
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*(2**((beta-1)/2))))**(1/beta)
    levy_step = 0.01 * (1 + Best_score/np.linalg.norm(Positions))*sigma/(np.abs(np.random.randn(SearchAgents_no,1))**beta)
    
    quantum_rot = np.exp(1j*np.random.rand(SearchAgents_no,dim)*np.pi)
    learn_prob = 0.4 + 0.5*np.abs(Best_score - np.linalg.norm(Positions-Best_pos,axis=1))/Best_score
    mask = np.random.rand(SearchAgents_no,dim) < learn_prob.reshape(-1,1)
    
    Positions = np.where(mask,
                        (Best_pos + levy_step*(quantum_rot.real*(Positions - Best_pos.mean(axis=0)))),
                        Positions*(1 + (np.random.rand(*Positions.shape)*0.72)**2 - 0.36))
    #EVOLVE-END       
    return Positions