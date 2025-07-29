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
    sigma = ((np.math.gamma(1+beta)*np.sin(np.pi*beta/2))/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    levy = 0.01*np.random.randn(SearchAgents_no,dim)*sigma / np.abs(np.random.randn(SearchAgents_no,dim))**(1/beta)
    
    norm_dist = np.linalg.norm(Positions-Best_pos,axis=1)
    learn_prob = 0.5 + 0.5*np.sin(np.pi*(Best_score-norm_dist)/(2*Best_score + 1e-12))
    
    perturbation = 0.1*(np.random.rand(*Positions.shape)-0.5)
    Positions = np.where(np.random.rand(*Positions.shape) < learn_prob.reshape(-1,1),
                        Best_pos * (1 - 0.2*levy) + 0.8*Positions + perturbation,
                        Positions + levy*rg)
    #EVOLVE-END       
    return Positions