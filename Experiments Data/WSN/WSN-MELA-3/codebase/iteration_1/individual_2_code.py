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
    # Adaptive exploration-exploitation balance
    exploration_prob = 0.5 * (1 + np.cos(np.pi * rg / 2))
    if np.random.rand() < exploration_prob:
        # Exploration phase (Levy flights)
        step = np.random.standard_cauchy(size=Positions.shape) * rg
        Positions += step * (Best_pos - Positions)
    else:
        # Exploitation phase (local refinement)
        gaussian_noise = 0.1 * rg * np.random.randn(*Positions.shape)
        Positions = Best_pos + gaussian_noise * (ub_array - lb_array)
    #EVOLVE-END       

    return Positions