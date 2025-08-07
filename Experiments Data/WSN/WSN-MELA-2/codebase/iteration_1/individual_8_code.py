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
    u = np.random.randn(SearchAgents_no, dim)*sigma
    v = np.random.randn(SearchAgents_no, dim)
    step = u/abs(v)**(1/beta)
    levy = 0.01*step*(Positions - Best_pos)
    
    # Adaptive exploration-exploitation
    w = 0.9 - (0.9-0.4)*(1 - np.exp(-rg*10))
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Opposition-based learning for 20% of population
    mask = np.random.rand(SearchAgents_no) < 0.2
    if np.any(mask):
        Positions[mask] = lb_array[mask] + ub_array[mask] - Positions[mask]
    
    # Update positions
    Positions = w*Positions + r1*(Best_pos - Positions) + r2*levy
    #EVOLVE-END
    
    return Positions