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
    op_pos = np.clip(0.5*(Positions + 1 - Positions + 0.1*Best_pos),0,1)
    w = 0.8*np.exp(-3*(np.arange(SearchAgents_no)/SearchAgents_no)**2)
    Positions = w.reshape(-1,1)*Positions + (1-w.reshape(-1,1))*op_pos
    
    # Dynamic elite guidance
    elite = Best_pos + 0.05*(Best_pos-Positions[np.random.permutation(SearchAgents_no)])
    rw = 0.6+0.3*np.sin(np.linspace(0,np.pi,SearchAgents_no)).reshape(-1,1)
    Positions = rw*Positions + (1-rw)*elite
    
    # Self-adaptive mutation
    neighbor_indices = np.random.choice(SearchAgents_no, size=(SearchAgents_no,))
    p_mut = 0.5-0.4*np.arange(SearchAgents_no)/SearchAgents_no    
    mask = np.random.rand(*Positions.shape) < p_mut.reshape(-1,1)
    perturbation = rg*0.7*(Positions[neighbor_indices]-Positions) * np.random.randn(*Positions.shape)
    Positions = np.where(mask, np.clip(Positions+perturbation, 0, 1), Positions)
    #EVOLVE-END       
    return Positions