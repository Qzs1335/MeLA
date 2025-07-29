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
    step_size = 0.01*step*(Positions - Best_pos)
    
    # Adaptive exploration-exploitation
    r = np.random.rand()
    if r < 0.5:
        Positions += step_size * np.random.randn(*Positions.shape)
    else:
        Positions = 0.5*(Positions + Best_pos + step_size)
    #EVOLVE-END

    return Positions