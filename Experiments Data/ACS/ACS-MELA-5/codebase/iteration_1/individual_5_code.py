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
    # Opposition-based learning component
    opposite_pos = 1 - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < 0.5, opposite_pos, Positions)
    
    # Adaptive cosine perturbation
    t = np.linspace(0, 2*np.pi, 360)
    cos_wave = np.cos(t[np.random.randint(0, 360, dim)])
    perturbation = rg * (1 - np.exp(-Best_score)) * cos_wave
    
    # Dynamic exploration-exploitation
    if np.random.rand() < 0.5:
        Positions += perturbation * (Best_pos - Positions)
    else:
        best_mask = np.random.rand(SearchAgents_no, dim) < 0.3
        Positions[best_mask] = Best_pos[best_mask] + 0.1*np.random.randn(*Best_pos[best_mask].shape)
    #EVOLVE-END

    return Positions