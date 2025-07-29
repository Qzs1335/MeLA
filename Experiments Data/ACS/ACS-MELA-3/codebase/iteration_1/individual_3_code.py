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
    # Adaptive mutation with cosine-based step sizes
    t = 1 - (1/(1 + np.arange(SearchAgents_no)))
    mutation_factor = 0.1 * np.cos(0.5*np.pi*t[:,None]) + 0.3
    
    # Elite-guided exploration
    elite_guides = Best_pos * (1 - mutation_factor) 
    random_walk = mutation_factor * (ub_array - lb_array) * np.random.rand(*Positions.shape)
    
    # Position update with exploitation-explosion balance
    exploit_mask = np.random.rand(SearchAgents_no, dim) < 0.5
    Positions = np.where(exploit_mask, 
                         elite_guides + random_walk,
                         Positions * (1 + mutation_factor * np.random.randn(*Positions.shape)))
    #EVOLVE-END       

    return Positions