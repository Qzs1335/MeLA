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
    alpha = 2.0 - 2.0 * (1 - 1/(rg+1)) 
    cos_weights = np.random.rand(SearchAgents_no, dim) * 2 * np.pi
    perturb = alpha * (Best_pos - Positions) * np.cos(cos_weights)
    levy_step = 0.01 * np.random.randn(*Positions.shape) * (Positions - Best_pos)
    Positions = Positions * (1 - alpha) + (Best_pos + perturb) * alpha + levy_step * (Best_score < 0.01)
    #EVOLVE-END       

    return Positions