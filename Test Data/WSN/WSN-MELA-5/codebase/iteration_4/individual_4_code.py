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
    # Adaptive cosine weights
    w = 0.5*(1 + np.cos(np.pi*rg)) 
    
    # Fitness-proportional neighborhood
    fitness = np.linalg.norm(Positions - Best_pos, axis=1)
    probs = 1/(1+fitness)
    probs /= probs.sum()
    
    # Vectorized hybrid update
    neighbor_ids = np.random.choice(SearchAgents_no, size=(SearchAgents_no,3), p=probs)
    local_bests = Positions[neighbor_ids.argmin(axis=1)]
    
    r1, r2 = np.random.rand(2,SearchAgents_no,dim)
    cognitive = 1.7*r1*(Best_pos-Positions)
    social = 1.3*r2*(local_bests-Positions)
    Positions = w*Positions + cognitive + social
    #EVOLVE-END       

    return Positions