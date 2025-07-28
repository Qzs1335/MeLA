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
    beta = 1.5 + np.random.rand()*0.5  # Dynamic beta
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(*Positions.shape) * sigma
    v = np.random.randn(*Positions.shape)
    step = 0.01*(rg+0.1)*u/abs(v)**(1/beta)  # rg-scaled step
    
    # Non-linear adaptive weights
    w = 0.9*np.exp(-Best_score/500)  # Exponential decay
    
    # Balanced hybrid update
    r = np.random.rand(SearchAgents_no, 1)
    mask = r < (0.3 + 0.4*(1-rg))  # Dynamic threshold
    Positions = np.where(mask,
                        Best_pos + w*step*(1+np.random.rand(*Positions.shape)),
                        w*Positions + step*(Best_pos - Positions))
    #EVOLVE-END       
    return Positions