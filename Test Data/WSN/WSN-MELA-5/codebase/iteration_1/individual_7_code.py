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
    # Levy flight parameter
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(SearchAgents_no, dim) * sigma
    v = np.random.randn(SearchAgents_no, dim)
    step = u/abs(v)**(1/beta) * 0.01
    
    # Adaptive weights
    w = 0.9 - 0.5 * (1 - np.cos(np.pi * rg/2))
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Hybrid update
    exploit_mask = (np.random.rand(SearchAgents_no) < w).reshape(-1,1)
    Positions = np.where(exploit_mask,
                        Positions + step + r1*(Best_pos - Positions),
                        Positions + step + r2*(np.mean(Positions,axis=0) - Positions))
    #EVOLVE-END       

    return Positions