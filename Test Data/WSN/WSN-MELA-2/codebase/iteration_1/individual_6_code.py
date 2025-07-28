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
    # Levy flight parameter
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    
    # Exploration phase (Levy flights)
    levy = 0.01 * np.random.randn(SearchAgents_no, dim) * sigma / (np.abs(np.random.randn(SearchAgents_no, dim))**(1/beta))
    explore_mask = np.random.rand(SearchAgents_no, dim) < 0.5
    Positions = np.where(explore_mask, Positions * (1 + rg*levy), Positions)
    
    # Exploitation phase (local search)
    local_scale = 0.1 * rg * np.random.randn(SearchAgents_no, dim)
    Positions = np.where(~explore_mask, Best_pos + local_scale, Positions)
    #EVOLVE-END       

    return Positions