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
    chaotic_factor = 3.9 * rg * (1 - rg)  # Logistic map for chaos
    quantum_angle = np.pi * np.exp(-Best_score/1000)  # Quantum rotation
    
    r1 = np.random.rand(SearchAgents_no, 1)
    r2 = np.random.rand(SearchAgents_no, 1)
    exploit = Best_pos - chaotic_factor * quantum_angle * r1 * abs(Best_pos - Positions)
    explore = Positions + quantum_angle * r2 * (np.random.permutation(Positions) - Positions)
    
    mask = (np.random.rand(SearchAgents_no, dim) < np.exp(-Best_score/5000))
    Positions = np.where(mask, exploit, explore)
    #EVOLVE-END

    return Positions