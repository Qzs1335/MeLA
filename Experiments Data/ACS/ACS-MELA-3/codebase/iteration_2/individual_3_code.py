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
    # Enhanced opposition-based learning
    op_pos = lb_array + ub_array - Positions 
    w = 0.9 * np.exp(-np.arange(SearchAgents_no)/(SearchAgents_no/2))  # Exponential decay weights
    Positions = w.reshape(-1,1)*Positions + (1-w.reshape(-1,1))*op_pos
    
    # Elite-guided dimensional learning with perturbations
    D_dist = Best_pos - Positions + 0.01*rg*np.random.randn(*Positions.shape)
    alpha = 0.5*np.exp(-Best_score/(np.abs(Best_score)+1e-12))  # Adaptive learning rate
    dim_learn = Positions + alpha * D_dist * np.random.rand(*Positions.shape)
    
    # Dynamic probability application
    prob = 0.7 - 0.5*(Positions == Best_pos).all(axis=1)  # Lower prob near best
    mask = np.random.rand(*Positions.shape) < prob.reshape(-1,1)
    Positions = np.where(mask, dim_learn, Positions)
    #EVOLVE-END       
    
    return Positions