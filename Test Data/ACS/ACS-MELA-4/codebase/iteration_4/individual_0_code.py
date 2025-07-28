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
    # Optimized chaotic control with sigmoid decay
    decay = 1/(1 + np.exp(Best_score/10000 - 3))
    chaos = 4 * rg * decay * (1 - rg * decay)
    
    # Enhanced adaptive opposition learning
    opp_prob = 0.7 * np.exp(-0.02*Best_score/1000)
    opposite_pos = lb_array + ub_array - Positions
    mask = np.random.rand(SearchAgents_no, dim) < opp_prob
    Positions = np.where(mask, opposite_pos*(1-chaos) + Positions*chaos, Positions)
    
    # Elite-guided search with improved weights
    elite = Best_pos * (1 + chaos*np.random.randn(SearchAgents_no, dim))
    w = np.exp(-0.7*chaos*np.arange(1, SearchAgents_no+1)/SearchAgents_no).reshape(-1,1)
    Positions = elite*np.sqrt(w) + Positions*(1-np.sqrt(w))
    #EVOLVE-END
    
    return Positions