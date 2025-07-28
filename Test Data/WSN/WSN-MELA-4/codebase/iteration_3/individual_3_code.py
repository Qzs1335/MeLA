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
    # Hybrid PSO-Cosine-Levy strategy
    progress = np.clip(1 - (Best_score/1000), 0.1, 0.9)
    w = 0.9 * progress  # Dynamic inertia
    c1 = 1.5 * (1-progress)  # Cognitive
    c2 = 1.5 * progress  # Social
    
    # Elite-guided velocity with Levy component
    r1 = np.random.rand(*Positions.shape)
    r2 = np.random.rand(*Positions.shape)
    levy = np.random.randn(*Positions.shape) * (rg/(1+progress))
    velocity = w*levy + c1*r1*(Best_pos-Positions) + c2*r2*(Positions[np.random.permutation(SearchAgents_no)]-Positions)
    
    # Hybrid boundary handling
    new_pos = Positions + velocity*rg
    valid_mask = (new_pos >= lb_array) & (new_pos <= ub_array)
    Positions = np.where(valid_mask, new_pos, 
                        np.where(new_pos > ub_array, 
                                 Best_pos + 0.5*(ub_array-Best_pos)*np.random.rand(*Positions.shape),
                                 Best_pos - 0.5*(Best_pos-lb_array)*np.random.rand(*Positions.shape)))
    #EVOLVE-END       
    
    return Positions