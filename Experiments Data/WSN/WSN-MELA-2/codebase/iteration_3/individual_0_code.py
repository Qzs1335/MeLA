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
    # Enhanced Levy flight
    beta = 1.5 + 0.3*np.sin(Best_score/1000)  # Dynamic beta
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(*Positions.shape) * sigma * (0.1 + 0.9*rg)  # Range-scaled
    v = np.random.randn(*Positions.shape)
    step = u/abs(v)**(1/beta)
    
    # Nonlinear adaptive weights
    w = 0.9 / (1 + np.exp(0.002*(Best_score-800)))  # Sigmoid adaptation
    
    # Balanced hybrid update
    r = np.random.rand(SearchAgents_no, 1)
    mask = (r < 0.3 + 0.4*rg)  # Dynamic threshold
    Positions = np.where(mask,
                        Best_pos*(1-w) + w*(Positions + step),
                        Positions + w*(Best_pos - Positions)*np.random.normal(0,1,Positions.shape))
    #EVOLVE-END       
    return Positions