import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    #EVOLVE-START
    SearchAgents_no = Positions.shape[0]  # Get number of search agents from Positions
    
    # Enhanced opposition learning with adaptive weights
    op_pos = 1 - Positions
    cos_w = 0.7 + 0.3 * np.cos(np.pi * np.arange(SearchAgents_no)/SearchAgents_no)
    Positions = cos_w.reshape(-1,1)*Positions + (1-cos_w.reshape(-1,1))*op_pos
    
    # Elite-guided exponential decay
    decay = np.random.exponential(0.5, (SearchAgents_no,1))
    D_dist = (1+Best_score/1e12)*Best_pos - Positions*decay
    dim_jump = Positions + np.random.rand()*D_dist
    
    # Stochastic dimension perturbation
    mask = np.random.rand(*Positions.shape) < 0.4
    Positions = np.where(mask, dim_jump, Positions)
    
    # Probabilistic phase switch
    rand_choice = np.random.rand(SearchAgents_no,1)
    Positions *= np.where(rand_choice>0.5, 1.2, 0.8)
    #EVOLVE-END       
    return Positions