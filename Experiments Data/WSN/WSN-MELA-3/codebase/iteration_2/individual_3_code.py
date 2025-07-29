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
    # Dynamic opposition-based learning
    dyn_prob = 0.5 * (1 - np.tanh(rg))  # Sigmoid probability adjustment
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no, dim) < dyn_prob, opposite_pos, Positions)
    
    # Enhanced nonlinear convergence
    a = 2 * (1 - 1/(1+np.exp(-rg/2))) * np.ones((SearchAgents_no, 1))
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Elite-guided exploration
    elite_mask = np.random.rand(SearchAgents_no, 1) < 0.3
    D_elite = np.abs(Best_pos.reshape(1, -1) - Positions)
    A = (2*a*r1 - a) * (1 + elite_mask*0.5)  # Boost exploration near elite
    C = 2*r2
    D = np.abs(C*Best_pos.reshape(1, -1) - Positions)
    Positions = np.where(elite_mask, 
                        Positions + 0.5*D_elite,
                        Best_pos.reshape(1, -1) - A*D)
    #EVOLVE-END       

    return Positions