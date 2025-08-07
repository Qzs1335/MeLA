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
    progress = 1/(1+np.exp(-(1000-Best_score)/200))  # Sigmoid progress
    w = 0.9*(1-progress)  # Non-linear inertia decay
    c1 = 2.5*progress*np.cos(np.pi*progress/2)  # Cosine-adapted cognitive
    c2 = 2.5 - c1  # Dynamic social
    
    velocity = np.clip(w*np.random.randn(*Positions.shape) + 
               c1*np.random.rand(*Positions.shape)*(Best_pos-Positions) + 
               c2*np.random.rand(*Positions.shape)*(Positions[np.random.permutation(SearchAgents_no)]-Positions),
               -0.1*rg, 0.1*rg)
    
    Positions = Positions + velocity
    boundary_violation = (Positions<lb_array)|(Positions>ub_array)
    Positions = np.where(boundary_violation, 
                        np.where(np.random.rand(*Positions.shape)<0.7,
                                lb_array + (ub_array-lb_array)*np.random.rand(*Positions.shape),
                                2*(ub_array*(Positions>ub_array) + lb_array*(Positions<lb_array)) - Positions),
                        Positions)
    #EVOLVE-END       
    
    return Positions