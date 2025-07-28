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
    # Enhanced chaotic control
    chaos = 3.99 * rg * (1 - rg) * (1 - 0.5*np.random.rand())
    
    # Adaptive opposition with dimension-wise probability
    opp_prob = 0.8 - 0.4*(Best_score/10000)
    prob_matrix = opp_prob * (1 + 0.1*np.random.randn(SearchAgents_no, dim))
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < np.clip(prob_matrix,0,1), opposite_pos, Positions)
    
    # Elite guidance with exponential decay
    w = np.exp(-2*chaos*np.arange(1, SearchAgents_no+1)/SearchAgents_no).reshape(-1,1)
    elite = Best_pos * (1 + 0.5*chaos*np.random.randn(SearchAgents_no, dim))
    Positions = elite*w + Positions*(1-w)
    
    # Dimension-wise chaotic scaling
    scale = 0.8 + 0.4*chaos*np.random.rand(SearchAgents_no, dim)
    Positions *= np.clip(scale, 0.5, 1.5)
    #EVOLVE-END
    
    return Positions