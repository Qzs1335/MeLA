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
    # Enhanced opposition with sigmoid adaptation
    opp_prob = 1/(1+np.exp(5*(rg-0.5)))  
    opposite_pos = lb_array + ub_array - Positions
    mask = (opposite_pos != Positions) & (np.random.rand(*Positions.shape) < opp_prob)
    Positions = np.where(mask, opposite_pos, Positions)
    
    # Nonlinear convergence control
    a = 2/(1+np.exp(3*rg*np.linspace(0,1,SearchAgents_no))).reshape(-1,1)
    r1 = np.random.randn(SearchAgents_no, dim)
    r2 = np.random.rand(SearchAgents_no, dim)
    
    # Hybrid DE/PSO update
    F = 0.5*(1+np.random.rand())
    mutant = Positions + F*(Best_pos - Positions) + 0.1*(1-rg)*np.random.randn(*Positions.shape)
    cross_mask = np.random.rand(*Positions.shape) < 0.9
    Positions = np.where(cross_mask, mutant, Positions)
    #EVOLVE-END       

    return Positions