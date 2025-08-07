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
    f_min = Best_score
    norm_pbest = f_min / (4 + f_min)
    
    # Improved opposition learning
    op_pos = lb_array + ub_array - Positions
    s_w = 1/(1+np.exp(0.5*np.arange(SearchAgents_no)/SearchAgents_no))
    Positions = s_w[:,None]*Positions + (1-s_w[:,None])*op_pos
    
    # Elite dimensional learning with adaptive noise
    elite_mask = np.random.rand(*Positions.shape) < norm_pbest
    guide_dist = (Best_pos[None,:] - Positions)/SearchAgents_no
    guided_moves = Positions*(1+rg) + np.random.normal(0,0.2,Positions.shape)*guide_dist*(1+rg/np.sqrt(dim))
    Positions = np.where(elite_mask, guided_moves, Positions)
    #EVOLVE-END       
    
    return Positions