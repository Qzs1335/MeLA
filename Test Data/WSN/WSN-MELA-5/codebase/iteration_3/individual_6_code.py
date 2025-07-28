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
    # Adaptive hybrid weights
    w = 0.7 + 0.2*np.cos(rg*np.pi/2)
    cognitive_weight = 1.3 - 0.5*rg
    social_weight = 1.1 + 0.4*rg
    
    # Elite-guided dimensional update
    elite_dims = np.random.choice(dim, int(dim*0.7), replace=False)
    delta = Best_pos - Positions
    
    # Neighborhood search with reflection
    for i in range(SearchAgents_no):
        if np.random.rand() < 0.7:
            r1, r2 = np.random.rand(2)
            Positions[i] = w*Positions[i] + cognitive_weight*r1*delta[i] + social_weight*r2*(Positions.mean(0)-Positions[i])
        else:
            theta = np.random.rand()*2*np.pi
            Positions[i,elite_dims] = Best_pos[elite_dims] + rg*np.cos(theta)*np.abs(delta[i,elite_dims])
    #EVOLVE-END       

    return Positions