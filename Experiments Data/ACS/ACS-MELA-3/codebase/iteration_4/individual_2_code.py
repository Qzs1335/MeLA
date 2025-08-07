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
    # Dual-phase adaptive search
    progress = np.exp(-1.5*np.arange(SearchAgents_no)/SearchAgents_no).reshape(-1,1)
    a = rg*(0.5 + 0.5*np.sin(np.pi*np.random.rand(SearchAgents_no,1))) 
    
    # Elite-guided exploitation    
    scaling_factors = np.random.uniform(0, 1-a, (SearchAgents_no, dim))
    elite_guide = Best_pos[np.newaxis, :] + (Best_pos[np.newaxis, :] - Positions) * scaling_factors
    Positions = progress * elite_guide + (1-progress) * Positions
    
    # Neighborhood enhanced exploration    
    random_indices = np.array([np.random.permutation(SearchAgents_no) for _ in range(dim)]).T
    R = rg * (Positions[random_indices, np.arange(dim)] - Positions) * (1-progress)
    update_mask = np.random.rand(*Positions.shape) < a 
    Positions = np.where(update_mask, np.clip(Positions + R, lb_array, ub_array), Positions)
    #EVOLVE-END       
    return Positions