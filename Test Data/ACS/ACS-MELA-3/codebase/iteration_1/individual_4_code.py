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
    fitness_weights = 1/(1 + np.linalg.norm(Positions - Best_pos, axis=1).reshape(-1,1))
    elite_mask = (Positions == Best_pos).astype(float)
    rand_dim = np.random.randint(0, dim, (SearchAgents_no,1))
    dim_mask = np.eye(dim)[rand_dim.flatten()]
    
    G = 1/(1 + np.exp(-0.01*(6000 - Best_score))) 
    accelerations = G*fitness_weights*(Best_pos - Positions)
    Positions = (1-elite_mask)*(
        Positions + rg*(0.8*accelerations + 
        0.1*np.random.randn(*Positions.shape)*dim_mask + 
        0.1*np.random.randn(*Positions.shape))
    ) + elite_mask*(Best_pos + 0.4*rg*(2*np.random.rand(*Positions.shape)-1)*dim_mask)
    #EVOLVE-END       
    
    return Positions