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
    u = np.random.normal(0, sigma, (SearchAgents_no, dim))
    v = np.abs(np.random.normal(0, 1, (SearchAgents_no, dim)))
    levy_step = 0.01 * u / (v**(1/beta))
    
    q = np.random.uniform(0, 1, SearchAgents_no)
    phi = np.random.uniform(0, 1, (SearchAgents_no, dim))
    adaptive_factor = np.exp(-rg * np.arange(1, SearchAgents_no+1)/SearchAgents_no).reshape(-1,1)
    
    Positions = np.where(q.reshape(-1,1) > 0.5,
                        Best_pos + levy_step * adaptive_factor * (phi*Best_pos - Positions),
                        Best_pos + levy_step * adaptive_factor * (Positions - phi.mean()*Best_pos))
    #EVOLVE-END       
    return Positions