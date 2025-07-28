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
    # Hybrid PSO-cosine strategy with elite guidance
    progress = np.clip(1 - (Best_score/1000), 0.1, 0.9)
    cosine_factor = np.cos(progress*np.pi/2)
    
    # Adaptive velocity components
    cognitive = 1.5*(1-progress)*np.random.rand(*Positions.shape)
    social = 1.5*progress*np.random.rand(*Positions.shape)
    
    # Elite-guided position update
    elite_mask = np.random.rand(SearchAgents_no,dim) < 0.7*progress
    Positions = np.where(elite_mask,
        Best_pos*(1 + rg*cosine_factor*np.random.randn(*Positions.shape)),
        Positions + cognitive*(Best_pos-Positions) + social*(Positions[np.random.permutation(SearchAgents_no)]-Positions))
    
    # Hybrid boundary handling
    out_of_bounds = (Positions < lb_array) | (Positions > ub_array)
    Positions = np.where(out_of_bounds, 
        np.where(np.random.rand(*Positions.shape)<0.5, 
            lb_array + (ub_array-lb_array)*np.random.rand(*Positions.shape),
            np.clip(Positions, lb_array, ub_array)), 
        Positions)
    #EVOLVE-END       
    
    return Positions