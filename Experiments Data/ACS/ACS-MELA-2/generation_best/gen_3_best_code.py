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
    levy_step = rg * np.random.randn(SearchAgents_no,1) * sigma / (np.abs(np.random.randn(SearchAgents_no,1))**beta)
    
    dist_factor = np.linalg.norm(Positions-Best_pos,axis=1,keepdims=True)
    mask = np.random.rand(*Positions.shape) < (0.5*(1 + Best_score/(Best_score+dist_factor)))
    Positions = np.where(mask, 
                        Best_pos + (levy_step*Positions)/dist_factor,
                        Positions*(1 + rg*(np.random.rand(*Positions.shape)-0.5)))
    #EVOLVE-END       
    return Positions