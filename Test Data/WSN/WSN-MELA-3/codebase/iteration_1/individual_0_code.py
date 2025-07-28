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
    # Adaptive exploration rate based on iteration progress
    adaptive_rg = rg * (1 + np.sin(np.linspace(0, np.pi, SearchAgents_no))).reshape(-1,1)
    
    # Memory component using exponential moving average
    memory_weight = 0.7 + 0.3*np.random.rand(SearchAgents_no, dim)
    memory_component = memory_weight * Best_pos + (1-memory_weight) * Positions
    
    # Crossover with best and random agents
    crossover_mask = np.random.rand(*Positions.shape) < 0.4
    rand_agents = Positions[np.random.randint(0, SearchAgents_no, SearchAgents_no)]
    Positions = np.where(crossover_mask, 
                        (Best_pos + rand_agents)/2 + adaptive_rg*np.random.randn(*Positions.shape),
                        memory_component + adaptive_rg*np.random.randn(*Positions.shape))
    #EVOLVE-END       

    return Positions