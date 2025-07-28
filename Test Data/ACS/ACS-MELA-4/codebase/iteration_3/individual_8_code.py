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
    chaos = 4 * rg * (1 - rg) * (1 - 0.5*np.random.rand())
    
    # Dynamic opposition with adaptive threshold
    opp_thresh = 0.6 + 0.2*np.sin(Best_score/10000)
    opposite_pos = lb_array + ub_array - Positions
    use_opposite = (np.random.rand(SearchAgents_no, dim) < opp_thresh) & (Positions != Best_pos)
    Positions = np.where(use_opposite, opposite_pos*chaos, Positions)
    
    # Elite-guided search with adaptive decay
    w = np.exp(-2*chaos*np.linspace(0,1,SearchAgents_no)).reshape(-1,1)
    elite_guide = Best_pos*(1 + 0.1*np.random.randn(SearchAgents_no,dim))
    Positions = w*elite_guide + (1-w)*Positions
    
    # Chaotic dimension scaling
    scale = 0.8 + 0.4*chaos*np.random.rand(SearchAgents_no, dim)
    Positions = Positions * np.clip(scale, 0.5, 1.5)
    #EVOLVE-END
    
    return Positions