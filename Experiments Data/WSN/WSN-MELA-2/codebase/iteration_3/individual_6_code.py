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
    beta = 1.0 + rg  # Dynamic beta based on search progress
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(*Positions.shape) * sigma
    v = np.random.randn(*Positions.shape)
    step = np.clip(u/abs(v)**(1/beta), -1, 1)  # Bounded steps
    
    # Enhanced adaptive weights
    w = 0.9 - (0.4 * rg)  # Linear decay with search progress
    
    # Balanced hybrid update
    r = np.random.rand(SearchAgents_no, 1)
    mask = r < (0.3 + 0.2*rg)  # Dynamic threshold
    Positions = np.where(mask,
                        Best_pos + w*step*(Best_pos - Positions),
                        Positions + w*step*(Best_pos - Positions)*np.random.rand(*Positions.shape))
    #EVOLVE-END       
    return Positions