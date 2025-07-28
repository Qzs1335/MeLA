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
    alpha = 1 - np.exp(-0.01 * rg)  # Adaptive step decay
    levys = np.random.randn(SearchAgents_no, dim) * (np.random.rand(SearchAgents_no, dim) ** (-1/3))  # Levy flights
    gradient = Positions - Best_pos
    
    explorers = Positions + alpha * levys * (1 - 2 * np.random.randint(0, 2, dim))
    exploiters = Positions + alpha * (gradient + 0.1 * np.random.randn(*gradient.shape))
    
    mask = (np.arange(Positions.shape[0]) % 2 == 0)[:, None]
    Positions = np.where(mask, explorers, exploiters)
    #EVOLVE-END
    
    return Positions