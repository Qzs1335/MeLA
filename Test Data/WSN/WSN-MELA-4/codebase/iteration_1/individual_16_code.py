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
    adaptive_scale = np.exp(-Best_score/1000)  # Non-linear scaling
    rg_adaptive = rg * (0.5 + 0.5 * adaptive_scale)
    
    # Enhanced search patterns
    spiral = np.random.rand(SearchAgents_no, dim) * np.cos(2*np.pi*np.random.rand(SearchAgents_no, 1))
    levy = np.random.rand(SearchAgents_no, dim) * (Positions - Best_pos) / (1 + np.random.rand())
    
    mask = np.random.rand(SearchAgents_no, dim) < 0.5
    Positions = np.where(mask,
                        Best_pos + rg_adaptive * spiral,
                        Positions + rg_adaptive * levy)
    #EVOLVE-END       
    return Positions