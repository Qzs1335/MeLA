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
    # Enhanced chaotic map (logistic)
    chaos = 3.99 * rg * (1 - rg)
    
    # Dynamic opposition learning
    p_opposition = 0.3 + 0.4 * (1 - Best_score/np.max(Positions))
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < p_opposition, opposite_pos, Positions)
    
    # Elite-guided cosine search
    elite = Positions[np.argmin(np.linalg.norm(Positions, axis=1))]
    w = 0.4 * (1 + np.cos(np.pi * np.random.rand(SearchAgents_no, 1))) * chaos
    new_pos = (0.6*Best_pos + 0.4*elite) * (1 - w) + Positions * w
    
    # Adaptive scaling
    scale = 0.9 + 0.2 * chaos * np.random.rand(SearchAgents_no, dim)
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < 0.7, new_pos * scale, new_pos)
    #EVOLVE-END
    
    return Positions