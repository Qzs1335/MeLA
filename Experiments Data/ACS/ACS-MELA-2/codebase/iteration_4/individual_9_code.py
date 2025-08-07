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
    fitness = np.linalg.norm(Positions-Best_pos,axis=1)
    norm_fitness = fitness/np.max(fitness)
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*(2**((beta-1)/2))))**(1/beta)
    q = 0.15*(1 - Best_score/(Best_score+np.min(fitness)))
    levy_step = q * np.random.randn(SearchAgents_no,1)*sigma/(np.abs(np.random.randn(SearchAgents_no,1)+1e-12)**(1/beta))
    
    learn_prob = 0.6 - 0.3*norm_fitness
    mem_factor = 0.9*(1 - Best_score/(Best_score+np.median(fitness)))
    mask = np.random.rand(SearchAgents_no,dim) < learn_prob.reshape(-1,1)
    Positions = np.where(mask, 
                        mem_factor*Best_pos + levy_step*(Positions - Best_pos.mean(axis=0)), 
                        Positions*(1 + 0.1*(np.random.rand(*Positions.shape)+np.sqrt(norm_fitness.reshape(-1,1)))))
    #EVOLVE-END       
    return Positions