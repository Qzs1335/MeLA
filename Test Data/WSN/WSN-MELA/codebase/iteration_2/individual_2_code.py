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
    # Enhanced cosine-based perturbation
    theta = np.random.rand(SearchAgents_no, 1) * 2 * np.pi
    perturbation = np.cos(theta) * (1-rg) * (Best_pos - Positions) * np.random.randn(*Positions.shape)
    Positions += 0.7 * perturbation
    
    # Rank-based elite guidance
    ranks = np.argsort(np.linalg.norm(Positions - Best_pos, axis=1))
    elite_prob = 0.3 * rg * (ranks[:,None]/SearchAgents_no)
    elite_mask = np.random.rand(SearchAgents_no, dim) < elite_prob
    Positions = np.where(elite_mask, 
                        Best_pos + 0.5*rg*np.random.randn(*Positions.shape)*(1+Best_score), 
                        Positions)
    
    # Memory-enhanced boundary handling
    boundary_violation = (Positions < lb_array) | (Positions > ub_array)
    Positions = np.where(boundary_violation, 
                        0.5*(Best_pos + Positions) + 0.1*rg*np.random.randn(*Positions.shape), 
                        Positions)
    #EVOLVE-END       
    
    return Positions