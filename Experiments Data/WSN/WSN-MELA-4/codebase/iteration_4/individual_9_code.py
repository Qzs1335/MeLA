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
    # Enhanced adaptive parameters with nonlinear scaling
    progress = np.clip(1 - (Best_score / 1000), 0.1, 0.9)
    w = 0.9 * (1 - progress**0.5)
    c1 = 2.5 * (1 - progress**2)
    c2 = 2.5 - c1
    
    # Modified velocity update with elite guidance
    elite_idx = np.random.choice(SearchAgents_no, size=min(SearchAgents_no//4, SearchAgents_no), replace=False)
    elite_positions = Positions[elite_idx]
    
    # Repeat elite positions to match dimensions
    repeat_factor = int(np.ceil(SearchAgents_no / len(elite_positions)))
    repeated_elite = np.tile(elite_positions, (repeat_factor, 1))[:SearchAgents_no]
    
    velocity = w * np.random.randn(*Positions.shape) + \
              c1 * np.random.rand(*Positions.shape) * (Best_pos - Positions) + \
              c2 * np.random.rand(*Positions.shape) * (repeated_elite - Positions)
    
    # Smart boundary handling with probabilistic reflection
    Positions = Positions + velocity * (rg * (0.5 + np.random.rand()))
    mask = (Positions < lb_array) | (Positions > ub_array)
    Positions = np.where(mask, 
                        np.clip(Positions, lb_array, ub_array) + 
                        np.random.rand(*Positions.shape)*(Best_pos - Positions), 
                        Positions)
    #EVOLVE-END       
    
    return Positions