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
    # Velocity component
    velocity = np.random.randn(SearchAgents_no, dim) * 0.1
    
    # Adaptive coefficients
    w = 0.7 - (0.7-0.2)*(1-(rg/2.28))
    c1 = 1.5 * (rg/2.28)
    c2 = 1.5 * (1 - rg/2.28)
    
    # Update velocity and position
    r1, r2 = np.random.rand(2, SearchAgents_no, dim)
    velocity = w*velocity + c1*r1*(Best_pos - Positions) + c2*r2*(Best_pos[np.random.randint(SearchAgents_no)] - Positions)
    Positions = Positions + velocity * (1 - 0.9*(rg/2.28))
    #EVOLVE-END
    
    return Positions