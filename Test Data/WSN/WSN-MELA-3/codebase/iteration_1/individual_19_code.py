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
    # Adaptive neighborhood search with size limit
    neighborhood_size = min(SearchAgents_no, max(3, int(SearchAgents_no * rg)))
    if SearchAgents_no > 1:
        neighbor_indices = np.array([np.random.choice(SearchAgents_no, neighborhood_size, replace=False) for _ in range(SearchAgents_no)])
        local_bests = Positions[neighbor_indices].mean(axis=1)
    else:
        local_bests = Positions.copy()
    
    # Memory-guided exploration with safe ratio
    memory_ratio = 0.3 * (1 - rg) if SearchAgents_no > 0 else 0
    memory_mask = np.random.rand(SearchAgents_no, dim) < memory_ratio
    memory_guide = np.random.rand(SearchAgents_no, dim) * Best_pos[np.newaxis, :]
    
    # Position update
    Positions = (1-memory_mask)*((1-rg)*local_bests + rg*Positions) + memory_mask*memory_guide
    #EVOLVE-END       

    return Positions