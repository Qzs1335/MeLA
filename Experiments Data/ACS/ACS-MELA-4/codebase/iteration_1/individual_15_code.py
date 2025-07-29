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
    opp_pos = lb_array + ub_array - Positions
    combined = np.vstack((Positions, opp_pos))
    fitness = np.array([np.sum(x) for x in combined])  # Simplified fitness for illustration
    idx = np.argsort(fitness)[:SearchAgents_no]
    Positions = combined[idx]
    
    # Adaptive weights
    w = 0.9 - (0.5 * rg)
    r1 = np.random.rand()
    r2 = np.random.rand()
    Positions = w*Positions + r1*(Best_pos - Positions) + r2*(Positions.mean(axis=0) - Positions)
    
    # Memory mechanism
    if rg < 0.3:  # Late stage
        mutation = 0.1*(ub_array - lb_array)*np.random.randn(*Positions.shape)
        Positions = np.where(np.random.rand(*Positions.shape) < 0.1, Positions + mutation, Positions)
    #EVOLVE-END       
    
    return Positions