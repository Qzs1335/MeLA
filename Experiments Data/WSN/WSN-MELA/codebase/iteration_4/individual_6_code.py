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
    # Levy flight enhanced search with safeguards
    eps = 1e-10  # Small epsilon to prevent division by zero
    beta = np.clip(1.5 + (1 - rg), 1.1, 2.9)  # Constrained beta range
    
    # Stable sigma calculation
    gamma_val = np.math.gamma(1+beta)
    sin_val = np.sin(np.pi*beta/2)
    denominator = np.math.gamma((1+beta)/2)*beta*(2**((beta-1)/2))
    sigma = ((gamma_val*sin_val)/(denominator + eps))**(1/(beta + eps))
    
    # Safe levy calculation
    u = np.random.randn(*Positions.shape)
    v = np.random.randn(*Positions.shape)
    levy = 0.01*rg * u * sigma / (np.abs(v) + eps)**(1/(beta + eps))
    
    # Adaptive neighborhood search
    perturbation = (0.4 + 0.3*rg) * (Best_pos - Positions) * (1 + 0.5*np.random.randn(*Positions.shape))
    Positions += perturbation + levy
    
    # Dynamic boundary reflection
    out_of_bounds = (Positions < lb_array) | (Positions > ub_array)
    Positions = np.where(out_of_bounds, 
                        np.clip(2*lb_array - Positions, lb_array, ub_array) if np.random.rand() < 0.5 
                        else np.clip(2*ub_array - Positions, lb_array, ub_array),
                        Positions)
    #EVOLVE-END       
    
    return Positions