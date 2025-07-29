import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    * The rest remains unchanged. *
    #EVOLVE-START
    # Adaptive OBL with cosine decay
    op_pos = lb_array + ub_array - Positions
    w = 0.5*(1 + np.cos(np.pi*np.arange(SearchAgents_no)/SearchAgents_no)).reshape(-1,1)
    Positions = w*Positions + (1-w)*op_pos
    
    # Rank-based elite guidance
    fitness_ranks = np.argsort(np.linalg.norm(Positions-Best_pos,axis=1))
    elite = Best_pos + 0.05*(Best_pos-Positions[fitness_ranks])
    rw = np.linspace(0.7,0.9,SearchAgents_no).reshape(-1,1)
    Positions = rw*Positions + (1-rw)*elite
    
    # Dimension-scaled mutation
    dim_scale = 1/np.sqrt(dim)
    mask = np.random.rand(*Positions.shape) < (0.4*dim_scale)
    perturbation = rg*dim_scale*(Positions[neighbor_indices]-Positions)*np.random.randn(*Positions.shape)
    Positions = np.where(mask, np.clip(Positions+perturbation, 0, 1), Positions)
    #EVOLVE-END       
    return Positions