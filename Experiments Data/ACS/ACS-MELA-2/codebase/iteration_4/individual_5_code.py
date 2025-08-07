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
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*(2**((beta-1)/2))))**(1/beta)
    levy_step = rg/10 * np.random.randn(SearchAgents_no,dim) * sigma/(np.abs(np.random.randn(SearchAgents_no,dim))**beta)
    
    fitness = np.linalg.norm(Positions-Best_pos,axis=1)
    norm_fitness = (fitness-np.min(fitness))/(np.max(fitness)-np.min(fitness)+1e-15)
    learn_prob = 0.7 - 0.3*(1-norm_fitness**2)
    
    q = np.random.rand()
    inertia = 0.9*q if q < 0.5 else 1.1*q
    directed = Best_pos + levy_step*(1-np.log1p(fitness)).reshape(-1,1)
    random = Positions + inertia*(levy_step - 0.5)
    
    mask = np.random.rand(SearchAgents_no,dim) < learn_prob.reshape(-1,1)
    Positions = np.where(mask, directed, random)
    #EVOLVE-END       
    return Positions