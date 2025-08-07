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
    # Enhanced Levy flight with adaptive scaling
    beta = 1.5 + 0.5*np.sin(rg*np.pi/2)  # Dynamic beta
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    step = np.random.randn(*Positions.shape)*sigma / (abs(np.random.randn(*Positions.shape))**(1/beta))
    
    # Dynamic weight balancing
    w = 0.9*np.exp(-rg) + 0.1  # Exponential decay with exploration
    
    # Adaptive hybrid update
    r = np.random.rand(SearchAgents_no, 1)
    mask = r < (0.3 + 0.4*rg)  # Progressive exploitation
    Positions = np.where(mask,
                        Best_pos*(1-w) + w*(Positions + step),
                        Positions*(1-w) + (Best_pos + step)*w)
    #EVOLVE-END       
    return Positions