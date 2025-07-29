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
    u = np.random.randn(SearchAgents_no, dim) * sigma
    v = np.random.randn(SearchAgents_no, dim)
    step = u/abs(v)**(1/beta)
    levy_step = 0.01*step * (Positions - Best_pos)

    adaptive_rg = rg * (0.1 + 0.9*(1 - Best_score/1000))
    theta = np.random.rand(SearchAgents_no,1)*2*np.pi
    r = adaptive_rg * np.random.rand(SearchAgents_no,1)
    
    exploit_mask = (np.random.rand(SearchAgents_no,1) < 0.5).reshape(-1,1)
    Positions = np.where(exploit_mask,
                        Best_pos + r*np.cos(theta)*(Positions - Best_pos),
                        Positions + levy_step)
    #EVOLVE-END

    return Positions