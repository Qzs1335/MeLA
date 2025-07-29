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
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*(2**((beta-1)/2))))**(1/beta)
    levy = 0.01*sigma*np.random.randn(SearchAgents_no,dim)/(np.abs(np.random.randn(SearchAgents_no,dim)+1e-12)**(1/beta))
    
    current_min = np.min(np.linalg.norm(Positions-Best_pos,axis=1))
    prob = np.clip(0.5 + 0.3*(current_min/Best_score), 0.1, 0.9)
    mask = np.random.rand(SearchAgents_no,1) < prob.reshape(-1,1)
    
    exploration = Positions + levy*(Positions - Best_pos.mean(axis=0))
    exploitation = Best_pos + 0.1*(Positions - Best_pos)
    Positions = np.where(mask, np.where(rg>0.5, exploration, exploitation), Positions)
    #EVOLVE-END       
    return Positions