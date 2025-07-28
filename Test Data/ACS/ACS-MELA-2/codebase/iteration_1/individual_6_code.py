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
    # Levy flight implementation 
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    levy = 0.01 * np.random.randn(SearchAgents_no, dim) * sigma/(abs(np.random.randn(SearchAgents_no, dim))**(1/beta))
    
    # Local and global guidance
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    local_best = Positions[np.random.randint(0, SearchAgents_no, SearchAgents_no)]
    
    Positions += rg * levy + 0.1 * (r1*(Best_pos-Positions) + r2*(local_best-Positions))
    #EVOLVE-END       
    
    return Positions