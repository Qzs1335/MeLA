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
    # Adaptive parameters
    w = 0.7 * (1 - rg**2)  # Non-linear decay
    c1 = 1.8 * (1 - rg)    # Cognitive decay  
    c2 = 1.8 * rg          # Social growth
    
    # Vectorized neighborhood search
    neighbor_size = max(3, int(SearchAgents_no*(0.1 + 0.1*rg)))
    all_neighbors = np.array([np.random.choice(SearchAgents_no, neighbor_size, replace=False) for _ in range(SearchAgents_no)])
    
    # Calculate distances to Best_pos for all neighbors
    neighbor_positions = Positions[all_neighbors]
    distances = np.linalg.norm(neighbor_positions - Best_pos, axis=2)
    
    # Find best in each neighborhood
    best_in_neighborhood_indices = np.argmin(distances, axis=1)
    local_bests = neighbor_positions[np.arange(SearchAgents_no), best_in_neighborhood_indices]
    
    # Hybrid update with elite guidance
    r1, r2 = np.random.rand(2, SearchAgents_no, dim)
    Positions = w*Positions + c1*r1*(Best_pos-Positions) + c2*r2*(local_bests-Positions)
    #EVOLVE-END       

    return Positions