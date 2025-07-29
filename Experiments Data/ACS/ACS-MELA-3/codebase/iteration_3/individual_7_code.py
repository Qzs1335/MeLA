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
    # Temperature-based OBL 
    temp = 0.9*np.exp(-5*np.arange(SearchAgents_no)/SearchAgents_no)
    w = np.clip(temp + 0.1*np.random.randn(SearchAgents_no), 0.2, 0.9)
    Positions = w.reshape(-1,1)*Positions + (1-w.reshape(-1,1))*(1-Positions)
    
    # Adaptive elite guidance
    progress = 1 - (rg/2) # 0.5-1.5 range
    elite_mix = progress*(1.5-progress)
    elite = Best_pos + elite_mix*(Best_pos-Positions[np.random.permutation(SearchAgents_no)])
    
    # Fitness-aware mutation 
    rel_dist = np.linalg.norm(Positions-Best_pos,axis=1)/np.sqrt(dim)
    mut_prob = 0.1 + 0.4*(1-rel_dist)
    mask = np.random.rand(*Positions.shape) < mut_prob.reshape(-1,1)
    mutation = (ub_array-lb_array)*np.random.rand(*Positions.shape)*mut_prob.reshape(-1,1)
    Positions = np.where(mask, np.clip(Positions+mutation, lb_array, ub_array), Positions)
    #EVOLVE-END       
    return Positions