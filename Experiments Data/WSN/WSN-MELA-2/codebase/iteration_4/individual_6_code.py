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
    # Dynamic Levy flight
    beta = 1.5 + np.random.rand()*0.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(*Positions.shape) * sigma * rg
    v = np.random.randn(*Positions.shape)
    step = u/abs(v)**(1/beta)
    
    # Adaptive sigmoid weights
    w = 1/(1+np.exp(-Best_score/500))  # Sigmoid scaling
    
    # Elite-guided hybrid update
    r = np.random.rand(SearchAgents_no, 1)
    mask = r < (0.3 + 0.5*w)  # Dynamic threshold
    Positions = np.where(mask,
                        Best_pos*(1-w) + w*step*Positions,
                        Positions + (Best_pos - Positions)*np.random.normal(0, w, Positions.shape))
    
    # Boundary check
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), 
                        lb_array + (ub_array - lb_array)*np.random.rand(*Positions.shape), 
                        Positions)
    #EVOLVE-END       
    return Positions