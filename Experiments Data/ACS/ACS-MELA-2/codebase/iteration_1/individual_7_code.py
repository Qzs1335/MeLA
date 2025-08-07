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
    F = 0.8
    CR = 0.9
    
    # Differential evolution mutation
    mutant = Best_pos + F * (Positions[np.random.permutation(SearchAgents_no)] 
                            - Positions[np.random.permutation(SearchAgents_no)])
    
    # Crossover
    crossover_mask = np.random.rand(SearchAgents_no, dim) < CR
    trial_vectors = np.where(crossover_mask, mutant, Positions)
    
    # Adaptive selection and social component
    selection_mask = np.random.rand(SearchAgents_no) < rg
    Positions = np.where(selection_mask.reshape(-1,1), 
                        Best_pos - rg * (Best_pos - trial_vectors),
                        trial_vectors)
    #EVOLVE-END       
    return Positions