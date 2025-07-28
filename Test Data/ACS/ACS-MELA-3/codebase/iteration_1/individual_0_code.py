import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no, dim = Positions.shape
    
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))

    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(*Positions.shape)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)

    #EVOLVE-START
    # Chaotic mapping for exploration (Logistic map)
    chaos = 3.95 * Positions * (1 - Positions) * (1 + rg)
    
    # Elite guidance with adaptive weights
    w = 0.5 + np.random.rand()
    elite_guide = w * Best_pos * (1 + 0.1*np.random.randn(*Positions.shape))
    
    # Opposition-based exploration
    omega = Positions - 0.5*(Positions - np.mean(Positions, axis=0))
    
    # Adaptive combination
    decay = 0.9*(1 - rg)
    r = np.random.rand(*Positions.shape)
    Positions = decay*(r*chaos + (1-r)*elite_guide) + omega/dim
    # Positions should still satisfy constraint so...
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    #EVOLVE-END       

    return Positions