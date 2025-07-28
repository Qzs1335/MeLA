import numpy as np
import numpy as np
from numpy.random import rand

def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]

    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))

    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(*Positions.shape)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)

    #EVOLVE-START
    # Levy flight implementation
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = rand(*Positions.shape) * sigma
    v = rand(*Positions.shape)
    step = u/(abs(v)**(1/beta))
    
    # Opposition-based learning for best positions
    opposition = (ub_array + lb_array) - Best_pos
    r = rand()
    
    # Exponential decay for balance
    decay = np.exp(-0.01*rg)
    Positions = (r * decay * Positions + 
                 (1-r) * (Best_pos + step * (Positions - opposition)))
    #EVOLVE-END
    
    return Positions