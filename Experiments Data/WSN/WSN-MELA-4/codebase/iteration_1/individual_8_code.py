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
    w = 0.9 - (0.5 * rg)  # Dynamic inertia weight
    c1, c2 = 1.5, 2.0  # Cognitive and social factors
    velocity = np.random.randn(*Positions.shape) * 0.1
    
    r1 = np.random.rand(*Positions.shape)
    r2 = np.random.rand(*Positions.shape)
    personal_best = Positions.copy()
    
    cognitive = c1 * r1 * (personal_best - Positions)
    social = c2 * r2 * (Best_pos - Positions)
    velocity = w * velocity + cognitive + social
    Positions = Positions + velocity * (1 - rg)
    #EVOLVE-END
    
    return Positions