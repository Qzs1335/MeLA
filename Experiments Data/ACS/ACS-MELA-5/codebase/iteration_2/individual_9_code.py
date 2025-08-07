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
    F = 0.3 + 0.4 * np.cos(rg * np.pi/100)  # Cosine-adaptive scaling
    CR = 1 / (1 + np.exp(-0.1*(rg-50)))  # Sigmoid crossover rate
    
    elite_guidance = Best_pos * (0.5 + 0.5*np.random.rand())
    donor = Positions + F*(elite_guidance - Positions + Positions[np.random.permutation(SearchAgents_no)] - Positions)
    mask = (np.random.rand(*Positions.shape) < CR) | (np.random.rand(*Positions.shape) < 0.1)  # 10% forced exploration
    Positions = np.where(mask, donor, Positions)
    #EVOLVE-END       

    return Positions