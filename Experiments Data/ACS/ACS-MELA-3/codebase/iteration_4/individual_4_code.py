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
    # Adaptive OBL with sigmoid decay
    op_pos = lb_array + ub_array - Positions
    w = 0.5 + 0.4/(1+np.exp(5*(np.arange(SearchAgents_no)/SearchAgents_no-0.8)))
    Positions = w.reshape(-1,1)*Positions + (1-w.reshape(-1,1))*op_pos
    
    # Dynamic elite guidance
    permuted = Positions[np.random.permutation(SearchAgents_no)]
    rw = np.clip(0.5 + 0.3*np.sin(2*np.pi*np.random.rand(SearchAgents_no,1)), 0.1, 0.9)
    perturbation = 0.2 * (Best_pos - permuted) * np.random.rand(SearchAgents_no,dim)
    Positions = rw*Positions + (1-rw)*(Best_pos + perturbation)
    
    # Dimension-safe vector perturbation
    nei_mask = np.random.rand(*Positions.shape) < 0.4
    neighbor_shift = rg * (Positions[np.random.choice(SearchAgents_no,size=SearchAgents_no)]-Positions)
    Positions = np.where(nei_mask, np.clip(Positions+neighbor_shift,lb_array,ub_array), Positions)
    #EVOLVE-END       
    return Positions