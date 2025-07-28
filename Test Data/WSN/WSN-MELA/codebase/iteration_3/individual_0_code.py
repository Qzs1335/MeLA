import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]
    
    # Initialize boundary arrays with proper shape
    lb_array = np.zeros_like(Positions)
    ub_array = np.ones_like(Positions)
    
    # Boundary check and random adjustment
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(*Positions.shape)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)

    #EVOLVE-START
    # Reshape Best_pos to match Positions dimensions
    Best_pos_reshaped = np.tile(Best_pos, (SearchAgents_no, 1))
    
    # Hybrid PSO-GWO update
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    cognitive = 1.5 * r1 * (Best_pos_reshaped - Positions)
    social = 1.5 * r2 * (Best_pos_reshaped[np.random.permutation(SearchAgents_no)] - Positions)
    Positions += (cognitive + social) * (0.5 + 0.5*np.cos(np.pi*rg))
    
    # Adaptive elite guidance
    elite_prob = 0.3 - 0.2*rg
    elite_mask = np.random.rand(SearchAgents_no, dim) < elite_prob
    Positions = np.where(elite_mask, 
                        Best_pos_reshaped + rg*(ub_array-lb_array)*np.random.randn(*Positions.shape),
                        Positions)
    
    # Smart boundary handling
    over_ub = Positions > ub_array
    under_lb = Positions < lb_array
    Positions = np.where(over_ub | under_lb,
                        np.clip(Positions, lb_array, ub_array) + 
                        0.1*rg*np.random.randn(*Positions.shape),
                        Positions)
    #EVOLVE-END       
    
    return Positions