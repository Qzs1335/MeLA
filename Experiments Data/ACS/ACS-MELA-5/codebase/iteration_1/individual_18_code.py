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
    # Adaptive cosine search with opposition learning
    a = 2 * (1 - rg)  # Decreasing coefficient
    rand_pos = np.random.rand(SearchAgents_no, dim)
    opposition_pos = lb_array + ub_array - Positions
    
    # Dynamic exploration-exploitation
    A = 2 * a * np.random.rand(SearchAgents_no, 1) - a
    C = 2 * np.random.rand(SearchAgents_no, 1)
    p = 0.5 + 0.5 * rg  # Adaptive probability
    
    # Update positions
    cond = np.abs(A) < 1
    new_pos = np.where(cond,
                      Best_pos - A * np.abs(C * Best_pos - Positions),
                      opposition_pos - C * (rand_pos * opposition_pos - Positions))
    
    Positions = np.where(np.random.rand(SearchAgents_no, 1) < p, new_pos, Positions)
    #EVOLVE-END       
    return Positions