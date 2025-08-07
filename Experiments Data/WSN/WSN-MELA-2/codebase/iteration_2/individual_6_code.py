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
    beta = 1.5 + np.random.rand()*0.5  # Dynamic beta
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    step = np.random.randn(*Positions.shape)*sigma/np.abs(np.random.randn(*Positions.shape))**(1/beta)
    
    # Improved adaptive weights
    w = 0.9*(1 - rg) + 0.1  # Range-adaptive weight
    
    # Hybrid update with exploitation boost
    r = np.random.rand(SearchAgents_no, 1)
    mask = r < (0.6 - 0.4*rg)  # Dynamic threshold
    Positions = np.where(mask,
                        Best_pos + w*step*(1+np.exp(-Best_score)),  # Boost near convergence
                        w*Positions + (Best_pos - Positions)*np.random.rand(*Positions.shape))
    
    # Random restart
    restart_mask = np.random.rand(*Positions.shape) < 0.01*rg
    Positions = np.where(restart_mask, rand_adjust, Positions)
    #EVOLVE-END       
    return Positions