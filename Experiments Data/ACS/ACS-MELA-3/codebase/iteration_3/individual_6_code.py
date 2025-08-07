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
    # Adaptive OBL with dynamic decay 
    alpha = 0.5 * (1 + np.cos(np.pi*rg))  # Cyclic adaptation
    op_pos = Best_pos + alpha*(Best_pos - Positions)
    w = np.exp(-3*rg*(np.arange(SearchAgents_no)/SearchAgents_no)).reshape(-1,1)
    Positions = w*Positions + (1-w)*op_pos
    
    # Elite hybrid search with dynamic weights - FIXED DIMENSION ISSUE
    r1 = np.random.rand(SearchAgents_no,1)
    # Modified to handle dimensions properly
    delta_pos = Positions[np.random.randint(0, SearchAgents_no, SearchAgents_no)] - Positions
    elite = Best_pos + rg*(0.2 + 0.8*rg)*delta_pos
    rw = (0.4 + 0.6*r1*rg).reshape(-1,1)  # Progressive exploitation
    Positions = rw*elite + (1-rw)*Positions
    
    # Adaptive directional mutation
    pm = 0.3*(1 - 0.8*rg)  # Decaying mutation rate
    mask = np.random.rand(*Positions.shape) < pm
    # Modified neighbor selection to maintain shape
    neighbor_idx = np.array([np.random.choice(SearchAgents_no, size=dim) for _ in range(SearchAgents_no)])
    perturbation = rg*np.random.normal(0, 0.1, Positions.shape) * np.abs(Positions - Positions[neighbor_idx, np.arange(dim)])
    Positions = np.where(mask, np.clip(Positions+perturbation, 0, 1), Positions)
    #EVOLVE-END       
    
    return Positions