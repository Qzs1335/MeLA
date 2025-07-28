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
    # Enhanced chaotic map (logistic)
    chaos = 4 * rg * (1 - rg) * (1 - np.random.rand())
    
    # Dynamic opposition-based learning
    opp_prob = 0.7 - 0.3*(Best_score/10000)  # Adaptive probability
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < opp_prob, opposite_pos, Positions)
    
    # Elite-guided adaptive search
    elite = Best_pos * (1 + chaos*np.random.randn(SearchAgents_no, dim))
    w = np.exp(-chaos*np.arange(1, SearchAgents_no+1)/SearchAgents_no).reshape(-1,1)
    new_pos = elite*w + Positions*(1-w)
    
    # Dimension-wise scaling
    scale = 0.9 + 0.2*chaos*np.random.rand(SearchAgents_no, dim)
    Positions = new_pos * scale
    #EVOLVE-END
    
    return Positions