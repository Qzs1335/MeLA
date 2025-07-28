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
    levy = 0.01 * np.random.randn(SearchAgents_no, dim) * sigma / (np.abs(np.random.randn(SearchAgents_no, dim)) ** beta)

    dist_ratio = np.linalg.norm(Positions-Best_pos, axis=1)/Best_score
    prob = 0.5 + 0.3 / (1 + np.exp(-2*dist_ratio))
    mask = np.random.rand(*Positions.shape) < prob[:, np.newaxis]
    
    weighted_mean = 0.7*Best_pos + 0.3*Positions.mean(axis=0)
    Positions = np.where(mask,  
                          weighted_mean + levy*(Positions - weighted_mean),
                          Positions*(1 + 0.3*np.random.randn(*Positions.shape)))
    #EVOLVE-END       
    return Positions