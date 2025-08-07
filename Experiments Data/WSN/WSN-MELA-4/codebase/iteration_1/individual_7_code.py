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
    levy = 0.01 * step * (Positions - Best_pos)
    
    # Adaptive weights
    w = 0.9 - (0.5 * (Best_score/1000))  # Scale based on fitness
    
    # Opposition-based learning for half population
    mask = np.random.rand(SearchAgents_no) > 0.5
    Positions[mask] = lb_array[mask] + ub_array[mask] - Positions[mask] + np.random.rand()*0.1
    
    # Combined update
    r1 = np.random.rand(*Positions.shape)
    r2 = np.random.rand(*Positions.shape)
    Positions = w*Positions + levy + r1*(Best_pos - Positions) + r2*(Positions.mean(axis=0) - Positions)
    #EVOLVE-END
    
    return Positions