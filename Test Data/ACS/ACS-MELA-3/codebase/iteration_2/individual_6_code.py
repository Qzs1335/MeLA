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
    # Enhanced opposition with adaptive noise
    op_pos = lb_array + ub_array - Positions + (ub_array-lb_array)*0.1*np.random.randn(*Positions.shape)
    w = 0.9*np.exp(-2*(np.arange(SearchAgents_no)/SearchAgents_no)**2)
    Positions = w.reshape(-1,1)*Positions + (1-w.reshape(-1,1))*op_pos
    
    # Score-aware dimensional learning
    D_dist = (np.random.rand()*1.5 + 0.5) * (Best_pos - Positions)
    step = (1-np.exp(-Best_score/abs(np.mean(Positions)+1e-12)))
    dim_learn = Positions + step * np.random.randn(*Positions.shape) * D_dist
    
    # Dynamic selection
    mask = np.random.rand(*Positions.shape) < 0.4 + 0.3*np.sin(rg*np.pi/2)
    Positions = np.where(mask, dim_learn, Positions)
    #EVOLVE-END       
    
    return Positions