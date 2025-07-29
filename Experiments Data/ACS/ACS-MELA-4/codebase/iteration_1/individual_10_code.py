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
    # Opposition-based learning
    opposites = 1 - Positions
    combined = np.vstack((Positions, opposites))
    fitness = np.array([Best_score]*SearchAgents_no + [Best_score*0.9]*SearchAgents_no)
    elite_idx = np.argpartition(fitness, SearchAgents_no)[:SearchAgents_no]
    Positions = combined[elite_idx]

    # Adaptive search with memory
    memory = Best_pos * (0.5 + 0.5*np.random.rand(dim))
    r1 = np.random.rand() * rg
    r2 = np.random.rand() * (1-rg)
    Positions = r1*Positions + r2*memory + (1-r1-r2)*np.random.rand(SearchAgents_no, dim)
    #EVOLVE-END       

    return Positions