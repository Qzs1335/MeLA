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
    t = 1 - (rg/2.28)  # Normalized iteration progress
    w = 0.4 + 0.6 * t  # Adaptive inertia weight
    
    # Elite-guided differential evolution
    elite_mask = np.random.rand(SearchAgents_no, dim) < 0.7
    F = 0.5 * (1 + np.random.randn(SearchAgents_no, dim))
    diff = Best_pos - Positions[np.random.permutation(SearchAgents_no)]
    Positions = np.where(elite_mask, Positions + w * (Best_pos - Positions) + F * diff, Positions)
    
    # Cosine-based local search
    theta = 2*np.pi*np.random.rand(SearchAgents_no,1)
    r = rg * (0.5 + 0.5*np.random.rand(SearchAgents_no,1))
    local_search = r * np.cos(theta) * (Positions - Best_pos)
    Positions = np.where(np.random.rand(SearchAgents_no,1) < 0.3, Positions + local_search, Positions)
    #EVOLVE-END       

    return Positions