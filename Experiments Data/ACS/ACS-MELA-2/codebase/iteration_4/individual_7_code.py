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
    u = np.random.randn(SearchAgents_no, dim)*sigma
    v = np.random.randn(SearchAgents_no, dim)
    levy_step = 0.01*rg*u/(np.abs(v)**(1/beta))
    
    fitness_ratio = np.linalg.norm(Positions-Best_pos,axis=1)/Best_score
    learn_prob = 0.5*fitness_ratio + 0.3*(1-fitness_ratio/rg)
    
    quantum_radius = rg*(0.1 + 0.9*np.exp(-5*np.abs(1-learn_prob)))
    mask = np.random.rand(SearchAgents_no,dim) < learn_prob.reshape(-1,1)
    Positions = np.where(mask,
                        Best_pos + quantum_radius.reshape(-1,1)*levy_step*(Positions-Best_pos.mean(axis=0)),
                        Positions + levy_step*np.random.uniform(-1,1,Positions.shape))
    #EVOLVE-END       
    return Positions