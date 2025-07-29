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
    # Lévy flight step size
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(*Positions.shape) * sigma
    v = np.random.randn(*Positions.shape)
    step = u/abs(v)**(1/beta) * 0.1
    
    # Adaptive weight
    w = 0.5 + np.random.rand()/2
    
    # Dimensional learning
    D_mask = np.random.rand(*Positions.shape) > 0.5
    Positions = np.where(D_mask,
                        w*Best_pos + step,
                        Positions * (1 + 0.1*np.random.randn(*Positions.shape)))
    #EVOLVE-END       
    
    return Positions