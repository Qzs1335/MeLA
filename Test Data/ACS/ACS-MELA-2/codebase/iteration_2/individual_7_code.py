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
    beta = 1.0
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*(2**((beta-1)/2))))**(1/beta)
    levy = 0.01 * np.random.randn(SearchAgents_no,1) * sigma / (np.abs(np.random.randn(SearchAgents_no,1))**beta)
    
    fitness_ratio = np.linalg.norm(Positions-Best_pos,axis=1)/Best_score
    adapt_rate = 0.4 + 0.3*np.exp(-2*fitness_ratio).reshape(-1,1)
    masks = (np.random.rand(*Positions.shape) < adapt_rate)
    
    exploit_move = 0.5*(Best_pos - Positions)
    Positions = np.where(masks,
                       Positions + levy*(exploit_move if np.random.rand()<0.7 else Positions*(np.random.rand()-0.5)),
                       Positions*(1.5 - 0.6*adapt_rate))
    #EVOLVE-END       
    return Positions