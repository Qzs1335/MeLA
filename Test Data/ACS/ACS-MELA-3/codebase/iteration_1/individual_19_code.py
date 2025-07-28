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
    memory_factor = 0.7 + 0.3 * np.exp(-0.1 * Best_score)  # Fitness-based adaptation
    grad_direction = np.random.randn(*Positions.shape) * 0.1  # Local search perturbation
    
    crossover_mask = np.random.rand(*Positions.shape) < 0.4
    elite_influence = Positions[np.argmin(np.linalg.norm(Positions - Best_pos, axis=1))]
    
    blend = memory_factor * (Best_pos - Positions) + (1-memory_factor) * grad_direction
    Positions = np.where(crossover_mask, 
                        Best_pos + rg * blend, 
                        elite_influence + 0.1 * (Positions - elite_influence))
    #EVOLVE-END       

    return Positions