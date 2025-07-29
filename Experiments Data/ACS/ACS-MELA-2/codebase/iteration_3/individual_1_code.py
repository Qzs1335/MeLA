import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    #EVOLVE-START
    SearchAgents_no = Positions.shape[0]  # Derive from input shape
    beta = 1.5 - 0.5*(Best_score/np.max(np.linalg.norm(Positions,axis=1)))  # Adaptive beta
    sigma = (np.math.gamma(1+beta)*np.sin(np.pi*beta/2)/(np.math.gamma((1+beta)/2)*beta*(2**((beta-1)/2))))**(1/beta)
    levy_step =(0.01*np.random.randn(SearchAgents_no,1)*sigma/np.abs(np.random.randn(SearchAgents_no,1))**(1/beta))
    
    learn_prob = 0.5 + 0.4*(Best_score - np.min(np.linalg.norm(Positions-Best_pos,axis=1)))/Best_score
    noise = levy_step*(Positions - Positions[np.random.permutation(SearchAgents_no),:])  # Enhanced diversity
    Positions = np.where(np.random.rand(*Positions.shape)<learn_prob.reshape(-1,1), 
                        Best_pos*(1-0.2*rg) + noise,
                        Positions + rg*(np.random.rand(*Positions.shape)-0.5)) 
    #EVOLVE-END       
    return Positions