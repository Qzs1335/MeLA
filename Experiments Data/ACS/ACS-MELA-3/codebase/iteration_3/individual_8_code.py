import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    * The rest remains unchanged. *
    #EVOLVE-START
    # Adaptive OBL with sigmoid weights
    op_pos = 1 - Positions
    w = 1/(1+np.exp(5*(0.5-np.arange(SearchAgents_no)/SearchAgents_no)))
    Positions = w.reshape(-1,1)*Positions + (1-w.reshape(-1,1))*op_pos
    
    # Dynamic elite guidance
    fitness_std = np.std([np.linalg.norm(p-Best_pos) for p in Positions])
    rw = 0.5*(1 + np.cos(np.pi*fitness_std/(fitness_std+1e-8)))
    elite = Best_pos + 0.1*(Best_pos-Positions)
    Positions = rw*Positions + (1-rw)*elite
    
    # Levy-enhanced mutation
    mask = np.random.rand(*Positions.shape) < (0.1 + 0.4*(1-rw))
    step = np.random.standard_normal(Positions.shape) * rg * 0.01
    neighbor_delta = Positions[np.random.permutation(SearchAgents_no)] - Positions
    Positions = np.where(mask, np.clip(Positions+step+0.7*neighbor_delta, 0, 1), Positions)
    #EVOLVE-END       
    return Positions