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
    # Levy flight component
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(*Positions.shape)*sigma
    v = np.random.randn(*Positions.shape)
    step = u/abs(v)**(1/beta)
    
    # Adaptive exploration/exploitation
    exploit_prob = 0.5 + 0.4*(1 - rg)
    if np.random.rand() < exploit_prob:
        Positions = Best_pos + 0.01*step*(Positions - Best_pos)
    else:
        scale = rg*(np.random.rand(SearchAgents_no, dim)-0.5)*2
        Positions = Positions + scale*abs(step)
    #EVOLVE-END
    
    return Positions