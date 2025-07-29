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
    # Opposition-based learning with dynamic weights
    op_pos = 1 - Positions 
    w = 0.9 - (0.9-0.2)*np.sqrt(np.arange(SearchAgents_no)/SearchAgents_no)
    Positions = w.reshape(-1,1)*Positions + (1-w.reshape(-1,1))*op_pos
    
    # Dimensional learning with elite guidance 
    D_dist = np.tile(Best_pos, (SearchAgents_no, 1)) - Positions
    dim_learn =  Positions + np.random.rand(*Positions.shape) * D_dist * np.exp(np.random.rand() - Best_score/np.max(Best_score+1e-12))
    
    # Apply with probability
    mask = np.random.rand(*Positions.shape) < 0.5
    Positions = np.where(mask, dim_learn, Positions)
    #EVOLVE-END       
    
    return Positions