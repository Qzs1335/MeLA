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
    u = np.random.randn(*Positions.shape) * sigma
    v = np.random.randn(*Positions.shape)
    step = 0.01*u/abs(v)**(1/beta)
    
    # Dynamic adaptive weights
    w = 0.9*(1 - Best_score/2000) + 0.1*np.random.rand()
    
    # Balanced hybrid update
    r1 = np.random.rand(SearchAgents_no, 1)
    r2 = np.random.rand(SearchAgents_no, 1)
    Positions = np.where(r1 < 0.7, 
                        Best_pos + w*step + (Best_pos - Positions)*r2,
                        Positions + w*step*r2)
    #EVOLVE-END       
    return Positions