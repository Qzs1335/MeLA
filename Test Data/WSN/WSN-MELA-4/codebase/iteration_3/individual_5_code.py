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
    # Hybrid strategy with adaptive weights
    progress = np.clip(1 - (Best_score / 1000), 0.1, 0.9)
    w = 0.5 * (1 + np.cos(np.pi * progress))  # Oscillating inertia
    c1 = 1.5 * progress
    c2 = 2.5 - c1
    
    # Elite guidance from top 10% agents
    elite_size = max(1, int(SearchAgents_no * 0.1))
    elite_indices = np.random.choice(SearchAgents_no, elite_size, replace=False)
    elite_guide = np.mean(Positions[elite_indices], axis=0)
    
    # Hybrid velocity update
    velocity = w * np.random.randn(*Positions.shape) + \
               c1 * np.random.rand() * (Best_pos - Positions) + \
               c2 * np.random.rand() * (elite_guide - Positions)
    
    # Enhanced boundary handling
    Positions = Positions + velocity * (rg * (0.5 + 0.5 * np.random.rand()))
    cross_upper = Positions > ub_array
    cross_lower = Positions < lb_array
    Positions = np.where(cross_upper | cross_lower, 
                        lb_array + np.random.rand(*Positions.shape) * (ub_array - lb_array),
                        Positions)
    #EVOLVE-END       
    
    return Positions