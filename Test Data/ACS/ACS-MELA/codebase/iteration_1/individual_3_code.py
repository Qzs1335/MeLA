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
    # Chaotic component for exploration
    chaotic_map = np.mod(Positions * 2.5, 1.0) 
    
    # Handle Best_pos whether it's 1D or 2D
    if Best_pos.ndim == 1:
        Best_pos = np.tile(Best_pos, (SearchAgents_no, 1))
    
    # Personal best memory component
    pers_best = np.random.rand(SearchAgents_no, dim) < 0.3
    memory_pos = np.zeros_like(Positions)
    memory_pos[pers_best] = Best_pos[pers_best]
    
    # Opposition-based learning
    opposite_pos = lb_array + ub_array - Positions
    use_opposite = np.random.rand(SearchAgents_no, dim) < 0.2
    
    Positions = np.where(use_opposite, opposite_pos, 
                       (1-chaotic_map)*Positions + chaotic_map*memory_pos)
    #EVOLVE-END       
    return Positions