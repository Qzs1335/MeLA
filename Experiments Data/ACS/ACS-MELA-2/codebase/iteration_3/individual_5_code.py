import numpy as np
import numpy as np 

def heuristics_v2(Positions, Best_pos, Best_score, rg):
    """
    Implements a modified particle swarm optimization with Levy flights and learning probability
    
    Args:
        Positions: Current positions of search agents (numpy array)
        Best_pos: Best position found so far (numpy array)
        Best_score: Fitness value of best position (float)
        rg: Random generator step size factor (float)
        
    Returns:
        Updated positions of search agents
    """
    #EVOLVE-START
    SearchAgents_no = Positions.shape[0]  # Derive number of search agents from input
    
    # Handle case where Best_score is zero to avoid division by zero
    if Best_score == 0:
        Best_score = 1e-15  # Small non-zero value
    
    # Calculate Levy flight step
    beta = 1.5
    gamma_val = np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    gamma_val_denom = np.math.gamma((1 + beta)/2) * beta * 2**((beta-1)/2)
    sigma = (gamma_val / gamma_val_denom)**(1/beta)
    
    # Generate Levy steps for all search agents
    rand_nums = np.random.randn(SearchAgents_no, Positions.shape[1])
    levy_step = 0.1 * sigma * rg * rand_nums / (np.abs(rand_nums)**(1/beta))
    
    # Calculate distance ratio with dimension handling
    dist_ratio = np.linalg.norm(Positions - Best_pos, axis=1, keepdims=True) / Best_score
    
    # Calculate learning probability
    learn_prob = 0.3 + 0.7 * np.exp(-5 * dist_ratio)
    
    # Update positions based on learning probability
    mask = np.random.rand(*Positions.shape) < learn_prob
    rand_pos = np.random.rand(*Positions.shape)
    
    Positions = np.where(mask,
                        Best_pos + levy_step * (Best_pos - Positions * rand_pos),
                        Positions * (0.9 + 0.2 * np.random.rand(*Positions.shape)))
    #EVOLVE-END
    
    return Positions