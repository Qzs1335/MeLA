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
    alpha = 0.5 * (1 + np.cos(np.pi * rg))  # Dynamic adaptation
    elite_mask = np.random.rand(SearchAgents_no) < alpha
    perturbation = np.random.normal(0, rg, (SearchAgents_no, dim))
    
    # Elite-guided exploration
    elite_guide = Best_pos[np.newaxis,:] + perturbation * (1 - alpha)
    random_explore = Positions + rg * np.random.randn(*Positions.shape)
    
    Positions = np.where(elite_mask[:,np.newaxis], 
                        alpha*elite_guide + (1-alpha)*random_explore,
                        Positions)
    #EVOLVE-END       
    return Positions