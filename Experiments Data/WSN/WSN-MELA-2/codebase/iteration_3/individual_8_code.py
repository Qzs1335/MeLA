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
    # Optimized Levy flight
    beta = 1.0 + rg  # Dynamic beta based on rg
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    step = np.random.randn(*Positions.shape)*sigma / (np.abs(np.random.randn(*Positions.shape))**(1/beta))
    
    # Enhanced adaptive weights
    w = 0.9 * (1 - np.tanh(Best_score/1000))  # Smoother scaling
    
    # Balanced hybrid update
    r1 = np.random.rand(SearchAgents_no, 1)
    r2 = np.random.rand(*Positions.shape)
    Positions = np.where(r1 < 0.5 + 0.3*rg,  # Dynamic threshold
                        Best_pos + w*step*(1-rg),
                        w*Positions + (Best_pos - Positions)*r2)
    
    # Boundary control
    Positions = np.clip(Positions, lb_array, ub_array)
    #EVOLVE-END       
    return Positions