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
    # Hybrid PSO-cosine strategy
    progress = 1 - (Best_score / 1000)
    w = 0.4 + 0.4 * progress**2  # Quadratic inertia adjustment
    c1 = 1.5 * np.cos(progress*np.pi/2)  # Cosine-based cognitive
    c2 = 2.5 - c1  # Complementary social
    
    # Enhanced velocity update
    cos_factor = np.cos(np.random.rand()*np.pi/2)
    velocity = w * np.random.randn(*Positions.shape) + \
               c1 * cos_factor * (Best_pos - Positions) + \
               c2 * (1-cos_factor) * (Positions[np.random.permutation(SearchAgents_no)] - Positions)
    
    # Adaptive boundary reflection
    Positions = Positions + velocity * rg
    reflect_ratio = 0.5 + 0.5*progress
    overflow = Positions > ub_array
    Positions = np.where(overflow, ub_array - reflect_ratio*(Positions-ub_array), Positions)
    underflow = Positions < lb_array
    Positions = np.where(underflow, lb_array + reflect_ratio*(lb_array-Positions), Positions)
    #EVOLVE-END       
    
    return Positions