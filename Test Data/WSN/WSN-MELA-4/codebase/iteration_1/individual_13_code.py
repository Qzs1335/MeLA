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
    # Dynamic parameters based on rg (decreases over iterations)
    w = 0.9 * rg  # Inertia weight
    c1 = 1.5 - rg  # Cognitive coefficient
    c2 = 0.5 + rg  # Social coefficient
    
    # Fitness-proportional perturbation
    perturbation = np.random.randn(*Positions.shape) * (0.1 + 0.9*(1-Best_score/1000))
    
    # Neighborhood search
    neighbors = np.random.choice(SearchAgents_no, size=(SearchAgents_no, 3), replace=True)
    local_best = np.mean(Positions[neighbors], axis=1)
    
    # Update equation
    r1, r2 = np.random.rand(2, SearchAgents_no, dim)
    vel = w * Positions + c1*r1*(Best_pos - Positions) + c2*r2*(local_best - Positions)
    Positions = Positions + vel + perturbation
    #EVOLVE-END
    
    return Positions