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
    # Adaptive OBL with fitness-aware decay
    beta = 1.5
    w = 0.9*np.exp(-beta*(np.arange(SearchAgents_no)/SearchAgents_no)**2)
    Positions = w.reshape(-1,1)*Positions + (1-w.reshape(-1,1))*(lb_array + ub_array - Positions)

    # Hierarchical leader guidance  
    rank = np.argsort(np.linalg.norm(Positions - Best_pos, axis=1))
    r1 = np.sin((1+rank/SearchAgents_no)*np.pi).reshape(-1,1)
    leader = Best_pos + 0.1*np.random.standard_cauchy((SearchAgents_no, dim)) 
    Positions = r1*Positions + (1-r1)*leader

    # Lévy mutation with adaptive probability
    step = rg*np.random.normal(0,1,Positions.shape)*np.abs(np.random.normal(0,1,Positions.shape))**(1/2.2)
    p_mut = 0.4*(1 - np.exp(-np.arange(SearchAgents_no)/20)).reshape(-1,1)
    mutate = np.random.rand(*Positions.shape) < p_mut
    Positions = np.where(mutate, np.clip(Positions+step, 0, 1), Positions)
    #EVOLVE-END       
    return Positions