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
    # Velocity-based PSO components
    if not hasattr(heuristics_v2, 'velocity'):
        heuristics_v2.velocity = np.zeros_like(Positions)
        heuristics_v2.pbest = Positions.copy()
        
    # Update personal bests
    mask = np.repeat((np.linalg.norm(Positions - Best_pos, axis=1) < 
                      np.linalg.norm(heuristics_v2.pbest - Best_pos, axis=1))[:, None], dim, axis=1)
    heuristics_v2.pbest = np.where(mask, Positions, heuristics_v2.pbest)
    
    # Combined update with crossover and mutation
    omega = 0.4 + 0.2 * np.random.rand()
    c1, c2 = 1.3, 1.7
    r1, r2 = np.random.rand(2, SearchAgents_no, dim)
    heuristics_v2.velocity = omega*heuristics_v2.velocity + c1*r1*(heuristics_v2.pbest-Positions) + c2*r2*(Best_pos-Positions)
    Positions += heuristics_v2.velocity * rg
    
    # Periodic restart if stagnating (track via global var)
    heuristics_v2.stagnation = getattr(heuristics_v2, 'stagnation', 0) + 1
    if heuristics_v2.stagnation > 50:
        Positions = np.random.rand(*Positions.shape)
        heuristics_v2.stagnation = 0
    #EVOLVE-END
    
    return Positions