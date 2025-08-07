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
    # Adaptive cosine weights
    theta = np.linspace(0, 2*np.pi, SearchAgents_no)
    w = 0.5*(np.cos(rg*theta)+1).reshape(-1,1)
    
    # Elite-guided local search
    elite_mask = (np.random.rand(SearchAgents_no) < 0.7)
    local_best = Positions[np.argmin(np.linalg.norm(Positions - Best_pos, axis=1))]
    pert = 0.1*rg*(np.random.randn(*Positions.shape))
    
    # Hybrid vector update
    r1, r2 = np.random.rand(2)
    cognitive = r1*(Best_pos - Positions)
    social = r2*(local_best - Positions)
    Positions = np.where(elite_mask[:,None], 
                        Positions + w*(cognitive + social + pert),
                        Positions + 0.5*w*(Best_pos - Positions))
    #EVOLVE-END       
    return Positions