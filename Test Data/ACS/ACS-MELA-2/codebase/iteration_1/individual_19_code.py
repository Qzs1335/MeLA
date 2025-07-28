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
    F = 0.8 * (1 - np.log(rg + 1)) # Adaptive scaling factor
    cauchy_noise = np.random.standard_cauchy(Positions.shape) * 0.1 * rg
    mask = np.random.rand(*Positions.shape) < 0.7  # Crossover probability
    Positions = np.where(mask,
                       Best_pos + F*(Positions[np.random.permutation(SearchAgents_no)] - Positions) + cauchy_noise,
                       Positions)
    #EVOLVE-END       
    return Positions