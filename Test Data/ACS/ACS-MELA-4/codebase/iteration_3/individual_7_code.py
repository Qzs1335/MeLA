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
    # Enhanced chaotic control parameter
    chaos = 3.9 * rg * (1 - rg) * (1 - 0.5*np.random.rand())
    
    # Adaptive opposition with elite influence
    opp_prob = 0.8 - 0.4*(Best_score/10000)
    opposite_pos = lb_array + ub_array - Positions + 0.1*chaos*(Best_pos - Positions)
    do_opposite = np.random.rand(SearchAgents_no, dim) < opp_prob
    Positions = np.where(do_opposite, opposite_pos, Positions)
    
    # Elite-guided search with dimension adaptation
    elite = Best_pos * (1 + 0.5*chaos*np.random.randn(SearchAgents_no, dim))
    w = np.exp(-2*chaos*np.arange(1, SearchAgents_no+1)/SearchAgents_no).reshape(-1,1)
    new_pos = elite*w + Positions*(1-w)
    
    # Dynamic dimension scaling
    scale = 0.8 + 0.4*chaos*np.random.rand(SearchAgents_no, dim)
    Positions = np.clip(new_pos * scale, lb_array, ub_array)
    #EVOLVE-END
    
    return Positions