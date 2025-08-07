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
    # Dynamic OBL with adaptive decay
    progress = np.exp(-8*(np.arange(SearchAgents_no)/SearchAgents_no)**2)
    op_pos = lb_array + ub_array - Positions
    Positions = progress.reshape(-1,1)*op_pos + (1-progress.reshape(-1,1))*Positions
    
    # Fitness-weighted elite guidance
    norm_score = Best_score/(Best_score+np.abs(Positions-Best_pos).mean(axis=1)+1e-8)
    elite = Best_pos + norm_score.reshape(-1,1)*(Best_pos-Positions[np.random.permutation(SearchAgents_no)])
    rw = np.sin(0.5*np.pi*norm_score).reshape(-1,1)
    Positions = rw*Positions + (1-rw)*elite
    
    # Progressive dimension-aware mutation
    dim_prob = 0.2 + 0.5*(np.arange(dim)/dim)*progress.mean() 
    mask = np.random.rand(*Positions.shape) < dim_prob
    scaler = (1-np.tanh(np.abs(Positions-elite)))*rg
    perturbation = scaler * np.random.randn(*Positions.shape)
    Positions = np.clip(np.where(mask, Positions+perturbation, Positions), 0, 1)
    #EVOLVE-END       
    return Positions