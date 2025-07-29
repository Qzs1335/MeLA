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
    beta = 1.5 * (1 - rg)  # Dynamic scaling
    u = np.random.randn(SearchAgents_no, dim) * 0.6966
    v = np.random.randn(SearchAgents_no, dim)
    levy = 0.01 * u / (np.abs(v)**(1/beta))
    
    progress = rg / 2.0  # Current optimization progress
    learn_prob = 0.5 + 0.4*(1 - progress)  # Progress-based probability
    mask = (np.random.rand(*Positions.shape) < learn_prob).astype(float)
    
    Positions = mask * (Best_pos + levy*(Positions - Best_pos)) + (1-mask)*Positions*1.2
    #EVOLVE-END       
    return Positions