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
    # Adaptive opposition weights
    op_pos = lb_array + ub_array - Positions
    sig_w = 0.8/(1+np.exp(-5*np.arange(SearchAgents_no)/SearchAgents_no))
    Positions = sig_w.reshape(-1,1)*Positions + (1-sig_w.reshape(-1,1))*op_pos
    
    # Elite-dimensional learning with nonlinear scaling
    scale = 0.5*np.cos(np.pi*np.random.rand()/2)+0.5
    D_dist = (np.tile(Best_pos,(SearchAgents_no,1))-Positions)*np.random.rand()**scale
    dim_learn = Positions + D_dist*np.exp(2*(np.random.rand()-Best_score/(Best_score+1)))
    
    # Adaptive mask with random restart  
    restart_mask = np.random.rand(*Positions.shape)<0.1
    p_mask = 0.6-0.4*np.sin(np.linspace(0,np.pi/2,SearchAgents_no))
    mutate_mask = np.random.rand(*Positions.shape)<p_mask.reshape(-1,1)
    Positions = np.where(restart_mask,np.random.rand(*Positions.shape),
                np.where(mutate_mask,dim_learn,Positions))
    #EVOLVE-END       
    
    return Positions