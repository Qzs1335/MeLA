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
    # Adaptive OBL with logistic decay
    op_pos = 1 - Positions
    w = 1/(1+np.exp(3-6*np.arange(SearchAgents_no)/SearchAgents_no))
    Positions = np.clip(w.reshape(-1,1)*Positions + (1-w.reshape(-1,1))*op_pos, 0, 1)
    
    # Dynamic elite hybridization
    elite = Best_pos + (0.05+rg/4)*(Best_pos-Positions[np.random.permutation(SearchAgents_no)])
    rw = (0.4 + 0.2*np.cos(np.pi*np.random.rand())).reshape(-1,1)
    Positions = rw*Positions + (1-rw)*elite
    
    # Adaptive mutation
    p_mut = np.clip(0.4 - 0.3*(Best_score/10000), 0.1, 0.4)
    mask = np.random.rand(*Positions.shape) < p_mut
    perturbation = (0.1+0.9*rg)*np.tan(np.pi*(np.random.rand(*Positions.shape)-0.5))
    Positions = np.where(mask, np.clip(Positions+perturbation, 0, 1), Positions)
    #EVOLVE-END       
    return Positions