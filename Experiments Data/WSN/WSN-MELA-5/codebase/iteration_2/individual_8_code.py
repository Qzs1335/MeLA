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
    w = 0.9 * (1 - rg**2)  # Nonlinear decay
    
    neighbor_size = max(3, int(SearchAgents_no*0.25))
    for i in range(SearchAgents_no):
        # Fitness-proportional neighbor selection with safety checks
        distances = np.linalg.norm(Positions - Best_pos, axis=1)
        if np.all(distances == 0):  # All agents at same position
            neighbors = np.random.choice(SearchAgents_no, neighbor_size)
        else:
            normalized = distances / (distances.mean() + 1e-10)  # Add small epsilon
            probs = np.exp(-normalized) + 1e-10  # Ensure non-zero probabilities
            probs = probs / probs.sum()  # Normalize
            neighbors = np.random.choice(SearchAgents_no, neighbor_size, p=probs)
        
        # Elite-guided update
        r1, r2 = np.random.rand(2)
        c1 = 2 * (1 - rg)  # Adaptive cognitive
        c2 = 2 * rg        # Adaptive social
        cognitive = c1 * r1 * (Best_pos - Positions[i])
        social = c2 * r2 * (Positions[neighbors].mean(axis=0) - Positions[i])
        Positions[i] = w*Positions[i] + cognitive + social + 0.1*(Best_pos-Positions[i])
    #EVOLVE-END       

    return Positions