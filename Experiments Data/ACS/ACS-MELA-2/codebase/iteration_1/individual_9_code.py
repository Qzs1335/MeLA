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
    t = 2 * np.random.rand() - 1  # Random weight [-1,1]
    spiral = 2 * np.random.rand(SearchAgents_no, dim) * np.exp(-np.random.rand() * t) 
    directional = (3 + np.random.rand(SearchAgents_no,1)) * Best_pos - spiral * Positions
    random_walk = 0.2 * np.random.randn(SearchAgents_no, dim) * (ub_array - lb_array)
    Positions = np.where(t > 0, spiral * np.cos(t) + directional, random_walk + Best_pos)
    #EVOLVE-END

    return Positions