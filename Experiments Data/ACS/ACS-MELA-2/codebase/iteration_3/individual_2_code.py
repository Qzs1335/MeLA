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
    u = np.random.randn(SearchAgents_no,dim)*0.745
    v = np.random.randn(SearchAgents_no,dim)
    levy_step = 0.01*u/(np.abs(v)**(1/beta))
    
    fitness_ratio = np.linalg.norm(Positions-Best_pos,axis=1)/Best_score
    adapt_prob = 0.5 + 0.4/(1+np.exp(-5*(fitness_ratio-np.mean(fitness_ratio))))
    mask = np.random.rand(SearchAgents_no,dim) < adapt_prob.reshape(-1,1)
    
    target_vec = Best_pos*(1-mask) + np.mean(Positions,axis=0)*mask
    Positions = target_vec + levy_step*(np.random.rand(SearchAgents_no,dim)-0.5)
    #EVOLVE-END       
    return Positions