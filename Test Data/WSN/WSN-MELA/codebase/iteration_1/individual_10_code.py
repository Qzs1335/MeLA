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
    # Adaptive inertia weight based on rg (decreases over iterations)
    w = 0.9 * (rg / 2.0) + 0.1
    
    # Cognitive and social components with nonlinear scaling
    c1 = 1.5 * (1 - np.exp(-Best_score/1000))
    c2 = 2.0 * np.exp(-Best_score/1000)
    
    # Velocity update with boundary reflection
    velocity = w * Positions + c1 * np.random.rand(*Positions.shape) * (Best_pos - Positions) \
               + c2 * np.random.rand(*Positions.shape) * (Best_pos.mean(axis=0) - Positions)
    
    # Position update with perturbation based on improvement rate
    perturbation = 0.1 * rg * (np.random.rand(*Positions.shape) - 0.5)
    Positions = Positions + velocity + perturbation
    
    # Reflective boundary handling
    Positions = np.where(Positions < lb_array, 2*lb_array-Positions, Positions)
    Positions = np.where(Positions > ub_array, 2*ub_array-Positions, Positions)
    #EVOLVE-END       

    return Positions