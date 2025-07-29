import numpy as np
import numpy as np 
def heuristics_v2(data_al, data_pb, Positions, Best_pos, Best_score, rg):
    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]

    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    # Dynamic boundary handling with memory
    past_best = Positions[np.random.choice(SearchAgents_no, 3, replace=False)]
    rand_adjust = np.where(np.random.rand(*Positions.shape) > 0.5,
                          Best_pos + rg*(past_best.mean(0) - Positions),
                          lb_array + (ub_array - lb_array)*np.random.rand(*Positions.shape))
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)

    #EVOLVE-START
    # Opposition-based exploration
    if np.random.rand() < 0.3:
        opposites = lb_array + ub_array - Positions
        Positions = np.vstack((Positions, opposites))[:SearchAgents_no]
    
    # Adaptive stepsize based on progress
    r = rg * (0.5 + (Best_score/(Best_score+1)) * np.random.randn(SearchAgents_no, 1))
    delta = Best_pos - Positions
    theta = 2*np.pi*np.random.rand(SearchAgents_no, dim)
    
    # Combination of exploitation and exploration
    Positions += r * (delta * np.cos(theta) + np.random.rand(*Positions.shape)**2)
    #EVOLVE-END
    
    return Positions