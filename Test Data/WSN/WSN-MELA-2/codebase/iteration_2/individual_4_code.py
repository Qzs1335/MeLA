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
    beta = 1.0 + rg  # Dynamic beta based on search radius
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(*Positions.shape) * sigma
    v = np.random.randn(*Positions.shape)
    step = u/abs(v)**(1/beta)
    
    # Enhanced adaptive weights
    w = 0.9 * (1 - rg)  # Weight decreases as search converges
    
    # Hybrid update with exploitation boost
    r = np.random.rand(SearchAgents_no, 1)
    mask = r < (0.3 + 0.5*rg)  # More exploitation as rg decreases
    Positions = np.where(mask,
                        Best_pos + w*step*Positions*(1+Best_score/1000),
                        w*Positions + (Best_pos - Positions)*np.random.rand(*Positions.shape)*(1+rg))
    
    # Random restart for diversity
    restart_mask = np.random.rand(*Positions.shape) < 0.01*rg
    Positions = np.where(restart_mask, rand_adjust, Positions)
    #EVOLVE-END       
    return Positions