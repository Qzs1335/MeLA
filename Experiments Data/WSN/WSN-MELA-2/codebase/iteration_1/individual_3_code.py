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
    # Levy flight component
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(*Positions.shape) * sigma
    v = np.random.randn(*Positions.shape)
    step = u/abs(v)**(1/beta)
    levy = 0.01 * step * (Positions - Best_pos)
    
    # Adaptive weights
    w = 0.9 * (1 - rg) + 0.1
    cosine_weight = np.cos(np.pi/2 * rg)
    
    # Hybrid update
    r = np.random.rand(SearchAgents_no, 1)
    Positions = np.where(r < 0.5,
                        w * Positions + cosine_weight * (Best_pos - Positions) + levy,
                        Positions * (1 + 0.5*rg*np.random.randn(*Positions.shape)))
    #EVOLVE-END
    
    return Positions