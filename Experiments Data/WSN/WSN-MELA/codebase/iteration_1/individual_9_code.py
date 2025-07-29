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
    w = 0.7 * (rg + 0.3)  # Adaptive inertia weight
    c1 = 2.0 * np.random.rand()
    c2 = 2.0 * np.random.rand()
    
    # Hybrid velocity update with rg adaptation
    cognitive = c1 * np.random.rand(*Positions.shape) * (Best_pos - Positions)
    social = c2 * np.random.rand(*Positions.shape) * (Best_pos[np.random.randint(SearchAgents_no)] - Positions)
    
    # Nonlinear rg scaling factor
    rg_factor = 1 - np.exp(-rg)
    Positions += w * (cognitive + social) * rg_factor
    
    # Periodic perturbation
    if np.random.rand() < 0.1:
        Positions += 0.1 * rg * np.random.randn(*Positions.shape)
    #EVOLVE-END

    return Positions