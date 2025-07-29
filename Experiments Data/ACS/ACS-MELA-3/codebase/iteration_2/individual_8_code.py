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
    # Enhanced OBL with exponential decay
    op_pos = 1 - Positions
    w = 0.9*np.exp(-2*(np.arange(SearchAgents_no)/SearchAgents_no))
    Positions = w.reshape(-1,1)*Positions + (1-w.reshape(-1,1))*op_pos
    
    # Elite-guided hybrid search
    elite = Best_pos + 0.1*(Best_pos-Positions[np.random.permutation(SearchAgents_no)])
    rw = (0.5*(1+np.sin(np.random.rand()*np.pi))).reshape(-1,1)
    Positions = rw*Positions + (1-rw)*elite
    
    # Probabilistic directional mutation 
    neighbor_indices = np.random.choice(SearchAgents_no, size=(SearchAgents_no,))
    mask = np.random.rand(*Positions.shape) < 0.3
    perturbation = rg * (Positions[neighbor_indices]-Positions) * np.random.rand(*Positions.shape)
    Positions = np.where(mask, np.clip(Positions+perturbation, 0, 1), Positions)
    #EVOLVE-END       
    return Positions