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
    # Enhanced cosine-based exploration
    theta = 2*np.pi*np.random.rand(SearchAgents_no, 1)
    cos_scale = np.cos(theta) * (1 - rg)  # Adaptive scaling
    
    # Dynamic elite guidance
    elite_prob = 0.2 * rg
    elite_mask = np.random.rand(SearchAgents_no, dim) < elite_prob
    elite_perturb = cos_scale * np.random.randn(*Positions.shape)
    Positions = np.where(elite_mask, Best_pos + elite_perturb, Positions)
    
    # Hybrid boundary handling
    boundary_violation = (Positions < lb_array) | (Positions > ub_array)
    reflection = np.where(Positions < lb_array, 2*lb_array-Positions, 2*ub_array-Positions)
    Positions = np.where(boundary_violation, 
                        reflection + 0.1*rg*np.random.randn(*Positions.shape),
                        Positions)
    #EVOLVE-END       
    
    return Positions