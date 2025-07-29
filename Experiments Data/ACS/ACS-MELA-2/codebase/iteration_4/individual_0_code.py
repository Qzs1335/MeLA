import numpy as np
import numpy as np 
def heuristics_v2(data_al, data_pb, Positions, Best_pos, Best_score, rg):
    * The rest remains unchanged. *
    #EVOLVE-START
    beta = 1.5
    conv_factor = 1 - (Best_score / Positions.shape[0])
    levy_scale = 0.01 * conv_factor
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    levy_step = levy_scale * np.random.randn(SearchAgents_no,1) * sigma/(np.abs(np.random.randn(SearchAgents_no,1))**beta/2)
    
    rel_fitness = 1 - (np.linalg.norm(Positions-Best_pos,axis=1)/Best_pos.max())
    learn_prob = np.clip(0.3 + 0.7*rel_fitness, 0.3, 0.95)
    
    mask = np.random.rand(SearchAgents_no,dim) < learn_prob.reshape(-1,1)
    quantum_spin = np.expand_dims(np.cos(2*np.pi*learn_prob),1)
    Positions = np.where(mask,
                       Best_pos + (quantum_spin*levy_step + 0.1*np.random.randn())*(Positions - Best_pos),
                       Positions*(1 + np.tanh(Best_score-np.linalg.norm(Positions,axis=1).mean()).reshape(-1,1)))
    #EVOLVE-END       
    return Positions