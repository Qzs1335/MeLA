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
    w = 0.9 - 0.5 * (1 - np.exp(-Best_score/1000))  # Nonlinear adaptive weight
    r1 = np.random.rand()
    r2 = np.random.rand()
    
    # Opposition-based learning
    if np.random.rand() < 0.3:
        opp_pos = 1 - Positions[np.random.randint(0, SearchAgents_no)]
        Positions += w * (opp_pos - Positions) * r1
    
    # Dynamic exploration/exploitation
    if rg > 0.5:  # Exploration phase
        levy = 0.01 * (np.random.randn(*Positions.shape) * (rg/10)) / np.power(abs(np.random.randn(*Positions.shape)), 1/3)
        Positions += levy
    else:  # Exploitation phase
        Positions = w * Positions + r2 * (Best_pos - Positions)
    #EVOLVE-END       

    return Positions