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
    # Enhanced Levy flight with dynamic scaling
    beta = 1.5 + np.random.rand()*0.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    step = rg * sigma * np.random.randn(*Positions.shape)/abs(np.random.randn(*Positions.shape))**(1/beta)
    
    # Sigmoid adaptive weights
    w = 1/(1+np.exp(-Best_score/500))  # Non-linear scaling
    
    # Elite-guided hybrid update with fixed dimensions
    r = np.random.rand(SearchAgents_no, 1)
    elite_mask = r < w.reshape(-1,1)
    
    # Create random indices for pairing while maintaining dimensions
    idx1 = np.random.permutation(SearchAgents_no)
    idx2 = np.random.permutation(SearchAgents_no)
    Positions = np.where(elite_mask,
                        Best_pos + step*(Positions - Best_pos),
                        Positions + step*(Positions[idx1] - Positions[idx2]))
    #EVOLVE-END       
    return Positions