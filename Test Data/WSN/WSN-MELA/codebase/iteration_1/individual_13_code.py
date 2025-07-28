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
    alpha = 2 * (1 - np.exp(-rg))  # Nonlinear convergence factor
    beta = np.random.rand(SearchAgents_no, dim)
    
    # Elite memory mechanism
    elite_mask = np.random.rand(SearchAgents_no) < 0.2
    Positions[elite_mask] = Best_pos * (1 + 0.1*(np.random.rand(sum(elite_mask), dim)-0.5))
    
    # Adaptive position update
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    Positions = alpha * Positions + (1-alpha) * (Best_pos + beta*(r1*Best_pos - r2*Positions))
    #EVOLVE-END       
    
    return Positions