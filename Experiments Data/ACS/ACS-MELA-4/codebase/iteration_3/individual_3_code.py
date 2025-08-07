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
    # Enhanced logistic chaotic mapping
    chaos = 4 * rg * (1 - rg)
    
    # Adaptive opposition learning
    opp_prob = 0.7 - 0.3*(Best_score/10000)
    mask = np.random.rand(SearchAgents_no, dim) < opp_prob
    Positions = np.where(mask, lb_array + ub_array - Positions, Positions)
    
    # Elite-guided search with adaptive weights
    w = np.exp(-np.linspace(0, 1, SearchAgents_no)*chaos).reshape(-1,1)
    elite = Best_pos * (1 + 0.1*np.random.randn(SearchAgents_no, dim))
    Positions = w*elite + (1-w)*Positions
    
    # Chaotic dimension scaling
    scale = 0.9 + 0.2*chaos*np.random.rand(SearchAgents_no, dim)
    Positions = Positions * scale
    #EVOLVE-END
    
    return Positions