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
    # Adaptive weight factor
    w = 0.9 - (0.9-0.4)*(rg%50)/50
    
    # Opposition-based learning
    if np.random.rand() < 0.3:
        opposite_pos = lb_array + ub_array - Positions
        Positions = np.where(np.random.rand(*Positions.shape)<0.5, opposite_pos, Positions)
    
    # Levy flight for exploration
    levy_step = np.random.randn(*Positions.shape) * np.power(np.abs(np.random.randn(*Positions.shape)), -1.5)
    Positions = np.where(np.random.rand(*Positions.shape)<0.1, Positions + 0.01*levy_step, Positions)
    
    # Memory-based update
    memory = np.random.rand(*Positions.shape) < 0.2
    Positions = np.where(memory, Best_pos + w*(Positions - Best_pos), Positions)
    #EVOLVE-END       

    return Positions