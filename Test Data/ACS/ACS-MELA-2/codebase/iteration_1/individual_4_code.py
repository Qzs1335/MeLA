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
    # Levy flight parameters
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    
    # Adaptive movement based on fitness
    w = 0.5 * (1 + np.sin(np.pi * np.random.rand()))
    c = 2.0 * (1 - (Best_score/(1000 + Best_score)))
    
    step = 0.01 * Best_pos * rg
    levy = sigma * np.random.randn(SearchAgents_no, dim) * (np.abs(np.random.randn(SearchAgents_no, dim))) ** (-1/beta)
    
    # Opposition-based learning
    oPositions = lb_array + ub_array - Positions
    opp_criterion = (np.random.rand(SearchAgents_no, dim) < 0.3)
    Positions = np.where(opp_criterion, np.where(oPositions < lb_array, lb_array, oPositions), Positions)
    
    # Momentum update
    v = w * Positions + c * (Best_pos[None,:] - Positions) * np.random.rand(SearchAgents_no, dim)
    Positions = Positions + rg * w * step + levy * v
    #EVOLVE-END       
    return Positions