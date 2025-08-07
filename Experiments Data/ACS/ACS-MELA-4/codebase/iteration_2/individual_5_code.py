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
    # Enhanced logistic chaotic map (now generates array for all agents)
    chaos = 4 * rg * (1 - rg) * (1 - 0.5*np.random.rand(SearchAgents_no, 1))
    
    # Dynamic opposition-based learning
    prob = 0.5 * (1 + np.sin(np.pi * rg))
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < prob, opposite_pos, Positions)
    
    # Elite-guided adaptive search
    elite = Best_pos * (1 + 0.1*np.random.randn(SearchAgents_no, dim))
    w = chaos * np.cos(np.pi * np.random.rand(SearchAgents_no, 1))
    new_pos = elite * (1 - w) + Positions * w
    
    # Dimension-wise scaling
    scale = 1 - (np.random.rand(SearchAgents_no, dim) * chaos)
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < 0.75, new_pos * scale, new_pos)
    #EVOLVE-END
    
    return Positions