import numpy as np
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]
    
    # Maintain boundary handling
    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))
    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(*Positions.shape)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)
    
    #EVOLVE-START
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*(2**((beta-1)/2))))**(1/beta)
    levy_step = 0.05*(rg/Best_score)*np.random.randn(SearchAgents_no,dim)*sigma/(np.abs(np.random.randn(SearchAgents_no,dim))**beta)
    
    scaling = np.linspace(0.1, 1, SearchAgents_no).reshape(-1,1)
    Positions = scaling*Best_pos + (1-scaling)*Positions
    Positions += levy_step*np.random.rand()
    #EVOLVE-END
    return Positions