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
    # Non-linear inertia weight
    w = 0.4 + 0.5/(1+np.exp(6*(rg-0.5)))
    
    # Vectorized neighborhood search
    neighbor_size = max(3, int(SearchAgents_no*0.2))
    all_neighbors = np.random.choice(SearchAgents_no, (SearchAgents_no, neighbor_size), replace=True)
    dists = np.linalg.norm(Positions[all_neighbors] - Best_pos, axis=2)
    probs = 1/(1+dists)
    selected = np.array([np.random.choice(neigh, p=p/p.sum()) for neigh,p in zip(all_neighbors,probs)])
    
    # Hybrid update with elite guidance
    r = np.random.rand(SearchAgents_no, 2)
    cognitive = 1.5 * r[:,0:1] * (Best_pos - Positions)
    social = 1.5 * r[:,1:2] * (Positions[selected] - Positions)
    Positions = w*Positions + cognitive + social + 0.1*(Best_pos-Positions)
    #EVOLVE-END       

    return Positions