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
    # Enhanced logistic chaotic map with adaptive scaling
    chaos = 3.99 * rg * (1 - rg) * (1 - 0.5*np.random.rand())
    
    # Dynamic opposition learning with nonlinear probability
    opp_prob = 0.8 * np.exp(-0.01*Best_score/1000)
    opposite_pos = lb_array + ub_array - Positions
    mask = np.random.rand(SearchAgents_no, dim) < opp_prob
    Positions = np.where(mask, opposite_pos*chaos, Positions)
    
    # Elite guidance with dimension-wise adaptive weights
    elite = Best_pos * (1 + 0.5*chaos*np.random.randn(SearchAgents_no, dim))
    w = np.exp(-2*chaos*np.arange(1, SearchAgents_no+1)/SearchAgents_no).reshape(-1,1)
    Positions = elite*w + Positions*(1-w) * (0.8 + 0.4*chaos*np.random.rand(SearchAgents_no, dim))
    #EVOLVE-END
    
    return Positions