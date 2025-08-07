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
    # Enhanced Levy flight
    beta = 1.5 + 0.5*np.random.rand()
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    step = np.random.randn(*Positions.shape)*sigma/np.abs(np.random.randn(*Positions.shape))**(1/beta)
    
    # Dynamic adaptive weights
    w = 0.9 - 0.4*(Best_score/1000) + 0.1*np.random.rand()
    
    # Balanced hybrid update
    r = np.random.rand(SearchAgents_no, 1)
    mask = r < (0.4 + 0.2*(1 - Best_score/1000))
    Positions = np.where(mask,
                        Best_pos + w*step*(1 + 0.1*np.random.randn(*Positions.shape)),
                        w*Positions + (Best_pos - Positions)*np.random.rand(*Positions.shape)*(1 + step))
    #EVOLVE-END       
    return Positions