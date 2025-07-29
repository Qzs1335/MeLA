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
    u = np.random.randn(*Positions.shape) * sigma
    v = np.random.randn(*Positions.shape)
    step = u/abs(v)**(1/beta)
    
    # Dynamic weight adjustment
    w = 0.9 - (0.9-0.4) * (np.arange(SearchAgents_no)/SearchAgents_no)
    
    # Hybrid update
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    cognitive = 1.5 * r1 * (Best_pos - Positions)
    social = 1.5 * r2 * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    Positions = w.reshape(-1,1)*Positions + cognitive + social + 0.01*step*(ub_array-lb_array)
    #EVOLVE-END
    
    return Positions