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
    beta = 1.5  # Levy exponent
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(SearchAgents_no, dim) * sigma
    v = np.random.randn(SearchAgents_no, dim)
    step = u/(np.abs(v)**(1/beta))
    levy_step = 0.01 * step * (Positions - Best_pos)
    
    momentum = 0.9 * (Positions - Best_pos) * (1 - rg)
    Positions = Positions + levy_step + momentum
    #EVOLVE-END
    
    return Positions