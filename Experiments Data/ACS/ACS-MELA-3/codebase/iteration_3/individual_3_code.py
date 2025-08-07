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
    # Enhanced OBL with adaptive weight
    op_pos = 1 - Positions
    omega = 0.9/(1+np.exp(-10*(Best_score-np.min(np.linalg.norm(Positions,axis=1)))/Best_score))
    w = omega + (1-omega)*np.random.rand(SearchAgents_no)
    Positions = w.reshape(-1,1)*Positions + (1-w.reshape(-1,1))*op_pos
    
    # Elite-guided search with cosine adaptivity
    theta = 0.5*np.pi*(1-np.exp(-rg))
    elite_vec = Best_pos - Positions[np.random.permutation(SearchAgents_no)]
    rw = (1 + np.sin(np.random.rand()*theta))/2
    Positions = rw.reshape(-1,1)*Positions + (1-rw.reshape(-1,1))*(Best_pos + elite_vec*np.random.rand())
    
    # Fitness-directed mutation 
    rel_fitness = 1.5 - np.linalg.norm(Positions-Best_pos,axis=1)/np.max(np.linalg.norm(Positions-Best_pos,axis=1))
    delta = Positions-np.mean(Positions,axis=0)
    mutant = Positions + rg*np.random.randn(*Positions.shape)*delta*(rel_fitness.reshape(-1,1))
    Positions = np.clip(mutant, lb_array, ub_array)
    #EVOLVE-END       
    return Positions