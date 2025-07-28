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
    fitness_variance = np.var([np.linalg.norm(pos-Best_pos) for pos in Positions])
    mutation_rate = 0.1*np.exp(-fitness_variance) 
    
    elite_mask = np.random.rand(SearchAgents_no) > 0.2
    mutated = Positions + mutation_rate*np.random.normal(0,1,Positions.shape)
    
    mate_mask = elite_mask[:,None] & (np.random.rand(*Positions.shape) < 0.7)
    Positions = np.where(mate_mask, 
                        (Positions + Positions[np.random.permutation(SearchAgents_no)])/2,
                        np.where(elite_mask[:,None], Best_pos + 0.1*(Positions-Best_pos), mutated))
    #EVOLVE-END
    
    return Positions