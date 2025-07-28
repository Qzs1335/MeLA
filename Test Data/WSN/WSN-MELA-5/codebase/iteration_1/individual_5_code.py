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
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(SearchAgents_no, dim)*sigma
    v = np.random.randn(SearchAgents_no, dim)
    step = u/abs(v)**(1/beta)
    
    exploitation_mask = np.random.rand(SearchAgents_no, dim) < 0.5
    exploration_step = 0.01*step*(Positions - Best_pos)
    exploitation_step = rg*(np.random.rand(SearchAgents_no, dim)*(Best_pos - Positions) + 
                          np.random.rand(SearchAgents_no, dim)*(Positions.mean(axis=0) - Positions))
    
    Positions = np.where(exploitation_mask, Positions + exploitation_step, Positions + exploration_step)
    #EVOLVE-END
    
    return Positions