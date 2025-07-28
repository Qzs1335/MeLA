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
    # Levy flight parameters
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    
    # Memory mechanism
    if np.random.rand() < 0.3:
        Best_pos = Best_pos * (0.5 + np.random.rand(*Best_pos.shape))
    
    # Dynamic weight adjustment
    w = 0.9 - (0.5 * (rg / 2.28))  # 2.28 is initial rg from history
    
    # Position update with levy flights
    u = np.random.randn(*Positions.shape) * sigma
    v = np.random.randn(*Positions.shape)
    step = u / (np.abs(v) ** (1/beta))
    
    Positions = w * Positions + (1-w) * Best_pos + 0.01 * step * (Best_pos - Positions)
    #EVOLVE-END       
    
    return Positions