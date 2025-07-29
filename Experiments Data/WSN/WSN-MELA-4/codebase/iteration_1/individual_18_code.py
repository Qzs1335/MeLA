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
    # Opposition-based learning for diversity
    opp_pos = 1 - Positions
    fitness = np.random.rand(SearchAgents_no, 1)  # Placeholder for actual fitness evaluation
    combined = np.vstack((Positions, opp_pos))
    combined_fitness = np.vstack((fitness, 1-fitness))
    elite_idx = np.argsort(combined_fitness.flatten())[:SearchAgents_no]
    Positions = combined[elite_idx]

    # Adaptive convergence factor
    a = 2 * (1 - rg**3)
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    A = 2*a*r1 - a
    C = 2*r2

    # Position update with adaptive parameters
    D = np.abs(C*Best_pos - Positions)
    Positions = Best_pos - A*D
    #EVOLVE-END

    return Positions