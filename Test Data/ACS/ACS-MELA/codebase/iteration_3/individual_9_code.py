import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    
    rand_adjust = lb_array + (ub_array - lb_array)*np.random.rand(SearchAgents_no,dim)
    Positions = np.where((Positions<lb_array)|(Positions>ub_array),rand_adjust, Positions)
    
    #EVOLVE-START
    T = np.exp(-np.linspace(0,10,SearchAgents_no))**3  # Sharper cooling 
    w = 1/(1+np.exp(-Best_score/T.mean())) * rg     # Sigmoid weighting
    
    phi = np.pi/3*np.random.rand(SearchAgents_no,dim) 
    R1 = np.abs(np.sin(phi))  # Trig-based randomization
    R2 = np.abs(np.cos(phi))
    
    cp_idx = np.random.permutation(SearchAgents_no)  # Single permutation per agent
    cp_idx = np.repeat(cp_idx[:, np.newaxis], dim, axis=1)  # Repeat for all dimensions
    
    explore_mask = (np.random.rand(SearchAgents_no,dim) > 0.3)  # Higher mutation
    exploit_term = Best_pos*(1+R1) - w*R2*Positions 
    explore_term = Positions[cp_idx, np.arange(dim)[np.newaxis,:]] + R1*np.tan(phi*T.reshape(-1,1))  
    
    Positions = np.where(explore_mask, explore_term, exploit_term)
    #EVOLVE-END
    
    return Positions