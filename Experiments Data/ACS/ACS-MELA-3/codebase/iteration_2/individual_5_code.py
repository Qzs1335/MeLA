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
    # Dynamic opposition learning with adaptive shaping
    op_pos = lb_array + ub_array - Positions
    w = np.clip(np.cos(np.pi*(np.arange(SearchAgents_no)/SearchAgents_no)/2)[::-1], 0.2, 0.9)
    sel = (np.random.rand(*Positions.shape) < w.reshape(-1,1))  # Fixed reshaping for proper broadcasting
    
    # Elite-guided dimension refinement
    beta = 1 - (0.5 * (Best_score/rg) if rg else 0)
    elite_vec = Best_pos - Positions
    cosine_sim = 1 + (np.einsum('ij,ij->i',Positions,elite_vec)/
                     (np.linalg.norm(Positions,axis=1)+1e-12))
    dim_learn = Positions + beta*(1-np.exp(-cosine_sim)).reshape(-1,1)*elite_vec
    
    ProbWeight = (0.4 + 0.3*np.exp(-np.arange(SearchAgents_no)/10)).reshape(-1,1)
    mask = np.random.rand(*Positions.shape) < ProbWeight
    Positions = np.where(mask, dim_learn, Positions)
    Positions = np.where(sel, op_pos, Positions)  # Moved after dimension refinement to maintain search diversity
    #EVOLVE-END       
    
    return Positions