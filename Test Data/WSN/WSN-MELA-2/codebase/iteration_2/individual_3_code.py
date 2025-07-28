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
    step = np.random.randn(*Positions.shape)*sigma / abs(np.random.randn(*Positions.shape))**(1/beta)
    
    # Dynamic adaptive weights
    w = 0.9*(1 - Best_score/2000) + 0.1*(rg/2.5)  # Dual scaling
    
    # Smart hybrid update
    r = np.random.rand(SearchAgents_no, 1)
    mask = r < (0.4 + 0.3*rg/2.5)  # Adaptive threshold
    Positions = np.where(mask,
                        Best_pos*(1-w) + w*(Positions + step),
                        Positions + w*(Best_pos - Positions)*np.random.rand(*Positions.shape))
    #EVOLVE-END       
    return Positions