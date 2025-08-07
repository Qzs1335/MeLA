import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    * The rest remains unchanged. *
    #EVOLVE-START
    # Adaptive OBL with progressive decay
    op_pos = lb_array + ub_array - Positions
    current_progress = 1 - np.mean(Best_score)/SearchAgents_no
    w = 0.95*np.exp(-3*current_progress*np.linspace(0,1,SearchAgents_no))
    Positions = w.reshape(-1,1)*Positions + (1-w.reshape(-1,1))*op_pos
    
    # Dynamic elite guidance
    elite = Best_pos + np.maximum(0.4, rg)*(Best_pos-Positions[np.random.permutation(SearchAgents_no)])*\
            (1 + 0.5*np.random.standard_cauchy(size=(SearchAgents_no,dim)))
    
    # Diversity-aware mutation
    diversity = np.mean(np.std(Positions, axis=0))
    mutation_prob = np.clip(0.4 - 0.3*diversity, 0.1, 0.35)
    mask = np.random.rand(*Positions.shape) < mutation_prob
    Positions = np.where(mask, np.clip(elite + 0.2*np.random.randn(*Positions.shape),0,1), Positions)
    #EVOLVE-END       
    return Positions