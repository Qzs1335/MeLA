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
    alpha = 0.5 + np.random.rand() * 0.5  # Adaptive coefficient
    cos_wave = np.cos(np.linspace(0, 2*np.pi, dim)).reshape(1, -1)
    elite_guide = Best_pos * (1 + np.random.randn(*Best_pos.shape)*0.1)
    rand_walk = np.random.randn(*Positions.shape) * rg
    
    Positions = alpha * (0.7*elite_guide + 0.3*Positions) \
                + (1-alpha) * (cos_wave * rand_walk)
    #EVOLVE-END       
    return Positions