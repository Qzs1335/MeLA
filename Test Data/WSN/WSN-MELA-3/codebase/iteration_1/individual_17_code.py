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
    # Opposition-based learning
    Opp_positions = lb_array + ub_array - Positions
    Candidates = np.vstack((Positions, Opp_positions))
    fitness = np.array([np.sum(pos) for pos in Candidates])  # Simplified fitness
    elite_idx = np.argmin(fitness)
    Positions = Candidates[elite_idx:elite_idx+SearchAgents_no]
    
    # Adaptive parameters
    w = 0.9 - (0.5 * rg)  # Dynamic inertia weight
    c1 = 2.0 * np.exp(-rg)  # Cognitive coefficient
    c2 = 2.0 - c1          # Social coefficient
    
    # Momentum-based update
    velocity = w * np.random.randn(*Positions.shape) + \
               c1 * np.random.rand() * (Best_pos - Positions) + \
               c2 * np.random.rand() * (np.mean(Positions, axis=0) - Positions)
    Positions = Positions + velocity * (1 - rg)
    #EVOLVE-END
    
    return Positions