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
    # Adaptive OBL with fast decay
    op_pos = np.clip(1 - Positions, 0, 1)
    w = 0.8*np.exp(-3*(np.arange(SearchAgents_no)/SearchAgents_no**0.5))
    Positions = w.reshape(-1,1)*Positions + (1-w.reshape(-1,1))*op_pos
    
    # Sinusoidal elite guidance
    elite = Best_pos + 0.15*(Positions[np.random.permutation(SearchAgents_no)]-Positions) 
    rw = np.abs(np.sin(np.random.rand(SearchAgents_no)*np.pi)).reshape(-1,1)
    Positions = (rw*Positions + (1-rw)*elite)
    
    # Normalized neighbor mutation 
    neighbor_idx = np.random.randint(0, SearchAgents_no, (SearchAgents_no,))
    mask = np.random.rand(*Positions.shape) < 0.25
    perturbation = 0.5*rg*(Positions[neighbor_idx]-Positions) * (np.random.rand(*Positions.shape)-0.5)
    Positions = np.where(mask, np.clip(Positions+perturbation, 0, 1), Positions)
    #EVOLVE-END       
    return Positions