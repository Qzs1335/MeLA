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
    # Adaptive OBL with cosine decay
    op_pos = lb_array + ub_array - Positions
    w = 0.6 + 0.3*np.cos(np.linspace(0,np.pi,SearchAgents_no))
    Positions = Positions*(1-w).reshape(-1,1) + op_pos*w.reshape(-1,1)
    
    # Elite-guided search
    rand_partner = Positions[np.random.permutation(SearchAgents_no)]
    sigmoid_rw = 1/(1+np.exp(-Best_score*np.random.rand(SearchAgents_no,1)*(1-np.linspace(0,1,SearchAgents_no).reshape(-1,1))))
    elite = Best_pos + 0.15*(Best_pos - rand_partner) 
    Positions = sigmoid_rw*Positions + (1-sigmoid_rw)*elite
    
    # Dimension-wise mutation
    scale_weights = np.random.rand(SearchAgents_no,dim)*np.exp(-5*np.linspace(0,1,dim))
    noise = scale_weights*np.random.randn(*Positions.shape)
    Positions = np.clip(Positions + rg*scale_weights*noise*(Best_pos-Positions), 0, 1)
    #EVOLVE-END       
    return Positions