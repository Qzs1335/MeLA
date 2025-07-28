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
    # Adaptive OBL with tuned decay
    op_pos = lb_array + ub_array - Positions
    w = 0.8*np.exp(-3*(np.arange(SearchAgents_no)/SearchAgents_no))
    Positions = np.where(np.random.rand(*Positions.shape)<0.7, w.reshape(-1,1)*Positions, op_pos)
    
    # Dynamic elite guidance
    t = 0.5*(1-np.cos(2*np.pi*np.arange(SearchAgents_no)/SearchAgents_no)).reshape(-1,1)
    elite = Best_pos + rg*(Best_pos - Positions[np.random.permutation(SearchAgents_no)])
    Positions = t*elite + (1-t)*Positions
    
    # Gradient-aware mutation
    grad_step = Best_score * (Positions - Positions[np.random.permutation(SearchAgents_no)])
    mut_prob = np.clip(0.4*(Best_score/(Best_score + np.linalg.norm(grad_step,axis=1))), 0.1,0.5)
    mask = np.random.rand(*Positions.shape) < mut_prob.reshape(-1,1)
    Positions = np.where(mask, np.clip(Positions+rg*grad_step*np.random.randn(*Positions.shape),0,1), Positions)
    #EVOLVE-END       
    return Positions