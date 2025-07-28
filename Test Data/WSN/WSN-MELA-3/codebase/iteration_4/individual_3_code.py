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
    # Enhanced dynamic opposition
    opp_prob = 0.5*(1 - 1/(1+np.exp(-5*(rg-0.5))))
    elite_mask = np.random.rand(SearchAgents_no,dim) < 0.1*rg
    opposite_pos = lb_array + ub_array - Positions
    Positions = np.where(np.random.rand(SearchAgents_no,dim)<opp_prob, 
                        opposite_pos*(1-elite_mask) + Best_pos*elite_mask, Positions)
    
    # Sigmoid adaptive factors
    a = 2/(1+np.exp(3*(rg-0.5)))
    r1 = np.random.rand(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Hybrid position update
    A = (2*a*r1 - a) * np.random.normal(0.5,0.2)
    C = 2*r2*(1 - 0.9*rg**2)
    leaders = np.where(np.random.rand(SearchAgents_no,1)<0.7*rg, 
                      Best_pos, Positions[np.random.permutation(SearchAgents_no)])
    D = np.abs(C*leaders - Positions)
    Positions += (Best_pos - A*D)*(0.5 + 0.5*rg) + 0.2*(1-rg)*np.random.randn(*Positions.shape)
    #EVOLVE-END       

    return Positions