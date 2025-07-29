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
    # Levy flight components for exploration
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    levy = 0.01*np.random.randn(SearchAgents_no, dim)*sigma/np.abs(np.random.randn(SearchAgents_no, dim))**(1/beta)

    # Adaptive exploration-exploitation balance 
    cosine_factors = np.cos(np.linspace(0, 2*np.pi, dim)) * np.log1p(Best_score) 
    exploit_prob = 0.3 + 0.6*(1 - np.exp(-0.005*rg)) 

    # Hybrid update rule
    leader_attraction = np.random.rand(SearchAgents_no, dim) < exploit_prob
    Positions = np.where(leader_attraction, 
                        cosine_factors*Best_pos + (1-cosine_factors)*Positions + np.random.rand()*levy,
                        Positions + levy*(Positions-np.roll(Positions, 1, axis=0)))
    #EVOLVE-END

    return Positions