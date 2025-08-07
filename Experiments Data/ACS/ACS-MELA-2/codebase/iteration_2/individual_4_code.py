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
    levy_step = 0.01*(Best_score+1e-6) * np.random.randn(*Positions.shape) * sigma / (np.abs(np.random.randn(*Positions.shape)+1e-6)**beta)
    direct_learn = 0.7 - 0.3*np.arange(SearchAgents_no)/SearchAgents_no
    Positions = np.where(np.random.rand(*Positions.shape)<direct_learn.reshape(-1,1), 
                        Best_pos + levy_step, 
                        Positions*(1 + (np.random.rand(*Positions.shape)-0.5)/SearchAgents_no))
    #EVOLVE-END       
    return Positions