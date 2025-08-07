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
    # Opposition-based learning
    opposition_pos = 1 - Positions
    fitness_diff = np.abs(Best_score - np.min([np.sum(Positions,axis=1), np.sum(opposition_pos,axis=1)], axis=0))
    Positions = np.where((fitness_diff < rg).reshape(-1,1), opposition_pos, Positions)

    # Adaptive nonlinear convergence
    t = 1 - (np.arange(SearchAgents_no)/SearchAgents_no
    w = 0.9 - 0.5*t
    c1 = 2.5 - 2*t
    c2 = 0.5 + 2*t

    # Elite guidance
    elite_mask = np.random.rand(SearchAgents_no) < 0.2
    elite_guide = Best_pos * (1 + 0.1*np.random.randn(*Best_pos.shape))
    Positions[elite_mask] = elite_guide[elite_mask]
    #EVOLVE-END

    return Positions