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
    # Lévy flight
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.normal(0, sigma, (SearchAgents_no, dim))
    v = np.random.normal(0, 1, (SearchAgents_no, dim))
    levy = 0.01*u / np.abs(v)**(1/beta)
    
    # Dynamic temperature
    max_iter = 50  # Assumed based on trials shown
    T = max(0.1, 1 - rg/max_iter)
    
    # Exp.decreasing rg and opposition learning
    new_rg = rg * 0.95
    opposite_pos = lb_array + ub_array - Positions
    replace_mask = (np.random.rand(SearchAgents_no, dim) < T)
    Positions = np.where(replace_mask, opposite_pos, Positions + levy)
    rg = new_rg
    #EVOLVE-END       
    return Positions