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
    # Adaptive opposition-based mutation
    cos_phase = np.random.rand() * 0.5 * np.pi
    adaptive_weight = (rg/5 + 0.1) * np.cos(cos_phase)
    op_pos = lb_array + ub_array - Positions
    Positions = adaptive_weight*Positions + (1-adaptive_weight)*op_pos
    
    # Elite-guided exponential perturbation
    elite_dist = np.abs(Best_pos - Positions) * np.exp(-0.8 * (rg + np.random.rand()))
    perturbation = elite_dist * (0.2 + np.random.rand(*Positions.shape)**2)
    mask = np.random.rand(*Positions.shape) < 0.8*np.exp(-2*rg)
    Positions = np.where(mask, Positions + perturbation, Positions)
    #EVOLVE-END       
    
    return Positions