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
    # Lévy flight implementation
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(SearchAgents_no, dim) * sigma
    v = np.random.randn(SearchAgents_no, dim)
    step = u/abs(v)**(1/beta) * 0.01
    
    # Adaptive scaling factor
    scale = rg * (1 - (Best_score/1000)) if Best_score < 1000 else rg*0.1
    
    # Opposition-based learning
    opp_pos = lb_array + ub_array - Positions
    select_mask = np.random.rand(SearchAgents_no, dim) < 0.5*rg
    Positions = np.where(select_mask, opp_pos, Positions + scale*step)
    #EVOLVE-END
    
    return Positions