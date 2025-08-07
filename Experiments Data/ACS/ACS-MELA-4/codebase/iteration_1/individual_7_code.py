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
    # Adaptive mutation with cosine waves
    t = np.linspace(0, 2*np.pi, dim)
    wave = np.cos(t * rg * (1 + Best_score/np.max(Positions))) 
    
    # Dynamic exploration/exploitation balance
    r1 = np.random.rand()
    r2 = np.random.rand()
    scale = 0.5 + 0.5*np.sin(rg*np.pi/2)
    
    # Position update
    mutation = wave * (Best_pos - Positions.mean(axis=0)) * r1
    neighborhood = Positions + scale * (Positions[np.random.permutation(SearchAgents_no)] - Positions) * r2
    Positions = np.where(np.random.rand(SearchAgents_no,dim)<0.5, Positions + mutation, neighborhood)
    #EVOLVE-END       
    
    return Positions