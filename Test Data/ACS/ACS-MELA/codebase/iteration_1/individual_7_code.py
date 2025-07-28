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
    # Adaptive cosine momentum
    theta = np.random.uniform(0, 2*np.pi, (SearchAgents_no, 1)) 
    A = 0.5*(1 + np.cos(2*np.pi*rg * np.arange(dim)/dim)).reshape(1,-1)
    
    # Lévy jump component
    beta = 1.5
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(beta*np.math.gamma((1+beta)/2)*2**((beta-1)/2)))**(1/beta)
    levy = 0.01*np.random.randn(SearchAgents_no,1) * sigma/np.abs(np.random.randn(SearchAgents_no,1))**(1/beta)
    
    Positions = np.where(np.random.rand(SearchAgents_no,1)<0.2,
                        Best_pos - A*np.cos(theta) + levy,
                        Positions + 0.2*(Best_pos - Positions)*(1+np.sin(rg*theta)))
    #EVOLVE-END       
    return Positions