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
    # Adaptive spiral motion control
    a = 2 - (rg * 2)  # Decreasing spiral coefficient
    r = np.random.rand(SearchAgents_no, dim)
    l = np.random.uniform(-1, 1, (SearchAgents_no, dim))
    q = np.random.rand(SearchAgents_no, dim)
    
    # Opposition-based learning for half the population
    mask = np.random.rand(SearchAgents_no) < 0.5
    if np.any(mask):
        Positions[mask] = lb_array[mask] + ub_array[mask] - Positions[mask]
    
    # Velocity-guided update
    velocity = 0.7 * Positions + 0.3 * Best_pos
    Positions = np.where(q < 0.5,
                        velocity * np.exp(a * r) * np.cos(2 * np.pi * l),
                        Best_pos + (Positions - Best_pos) * np.exp(a * r) * np.cos(2 * np.pi * l))
    #EVOLVE-END
    
    return Positions