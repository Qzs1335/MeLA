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
    # Enhanced chaotic mapping with adaptive parameter
    chaos = 3.99 * rg * (1 - rg) * (1 + np.random.rand()/10)
    
    # Improved opposition learning with nonlinear probability
    opp_prob = 0.8*np.exp(-Best_score/20000) + 0.1
    opposite_pos = lb_array + ub_array - Positions
    mask = np.random.rand(SearchAgents_no, dim) < opp_prob
    Positions = np.where(mask, opposite_pos*(1 + chaos*np.random.randn(*Positions.shape)), Positions)
    
    # Elite-guided search with exponential decay
    elite = Best_pos * (1 + 0.5*chaos*np.random.randn(SearchAgents_no, dim))
    w = np.exp(-2*chaos*np.linspace(0,1,SearchAgents_no)).reshape(-1,1)
    Positions = elite*w + Positions*(1-w)
    #EVOLVE-END
    
    return Positions