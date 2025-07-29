import numpy as np
import numpy as np
from scipy.special import gamma

def heuristics_v2(Positions, Best_pos, Best_score, rg):
    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]

    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))

    # Mirror reflection for bounds handling
    Positions = np.where(Positions < lb_array, -Positions, Positions)
    Positions = np.where(Positions > ub_array, 2*ub_array - Positions, Positions)

    #EVOLVE-START
    # Dynamic beta for Levy flights
    dyn_beta = 1.5 - 0.5*(1 - np.exp(-rg/10))
    sigma = (gamma(1+dyn_beta)*np.sin(np.pi*dyn_beta/2)/(gamma((1+dyn_beta)/2)*dyn_beta*(2**((dyn_beta-1)/2))))**(1/dyn_beta)
    theta = 2*np.pi*np.random.rand(SearchAgents_no,1)  # Bloch sphere rotation
    
    step = (np.linalg.norm(Positions - Best_pos, axis=1)/dim).reshape(-1,1)
    quantum_layer = np.exp(1j*theta)*step + step*(1-2*np.log(np.random.rand(SearchAgents_no,1)))
    levy_step = np.abs(quantum_layer.real) * sigma/np.abs(np.random.randn(SearchAgents_no,1)**dyn_beta)
    
    prob = np.linspace(0.7,0.3,SearchAgents_no)  # Rank-based
    mask = np.random.rand(SearchAgents_no,dim) < prob.reshape(-1,1)
    memory = 0.1*np.random.randn(*Positions.shape)  # Memory factor
    update = Best_pos + levy_step*(Positions - (0.9*Best_pos.mean(axis=0)+0.1*Positions.mean(axis=0)))
    Positions = np.where(mask, update + memory, Positions)
    #EVOLVE-END       
    return Positions