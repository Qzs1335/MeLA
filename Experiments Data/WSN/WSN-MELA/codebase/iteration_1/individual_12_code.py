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
    # Dynamic neighborhood search with adaptive weights
    w = 0.9 * (1 - np.exp(-5 * rg))  # Adaptive inertia weight
    cognitive = 1.5 * np.random.rand()
    social = 1.5 * np.random.rand()
    
    # Hybrid velocity update
    velocity = (w * Positions + 
               cognitive * np.random.rand(*Positions.shape) * (Best_pos - Positions) +
               social * np.random.rand(*Positions.shape) * (Best_pos[np.random.randint(SearchAgents_no)] - Positions))
    
    # Levy flight mutation for exploration
    levy_step = np.random.randn(*Positions.shape) * (rg ** 0.5)
    Positions = np.where(np.random.rand(*Positions.shape) < 0.1, 
                        Positions + levy_step, 
                        Positions + velocity)
    #EVOLVE-END       
    
    return Positions