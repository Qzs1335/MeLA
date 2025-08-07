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
    # Enhanced Levy flight with dynamic scaling
    beta = 1.5 + 0.3*np.sin(np.pi*rg/2)  # Dynamic beta
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    step = np.random.randn(*Positions.shape)*sigma / np.abs(np.random.randn(*Positions.shape))**(1/beta)
    
    # Sigmoid-based adaptive weights
    w = 1/(1+np.exp(-Best_score/500))  # Non-linear scaling
    
    # Elite-guided hybrid update
    r1 = np.random.rand(SearchAgents_no, 1)
    r2 = np.random.rand(SearchAgents_no, dim)
    Positions = np.where(r1<0.7, 
                        Best_pos*(1+w*step) + 0.1*r2*(ub_array-lb_array),
                        Positions + w*(Best_pos - Positions)*r2)
    #EVOLVE-END       
    return Positions