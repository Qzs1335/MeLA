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
    # Levy flight implementation
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(*Positions.shape)*sigma
    v = np.random.randn(*Positions.shape)
    step = u/abs(v)**(1/beta)
    
    # Adaptive weights
    w = 0.9 - 0.5 * (1 - 1/(1+np.exp(-Best_score/1000)))
    cos_weight = np.cos(np.pi/2 * (rg/10))
    
    # Opposition-based learning
    opposite_pos = lb_array + ub_array - Positions
    opp_mask = np.random.rand(*Positions.shape) < 0.1 * (1 - rg/10)
    
    Positions = np.where(opp_mask, opposite_pos, 
                        w * Positions + cos_weight * step * (Best_pos - Positions))
    #EVOLVE-END

    return Positions