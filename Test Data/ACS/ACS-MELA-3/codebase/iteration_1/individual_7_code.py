import numpy as np
import numpy as np 
def heuristics_v2(Positions, Best_pos, Best_score, rg):
    # Random selection and mutation factors
    k = np.random.randint(3)
    r1 = np.random.rand()
    F = 0.5 + r1 * rg * 0.5
    # Donor vector creation
    donor = Best_pos + F*(np.random.permutation(Positions)[k] - Positions)
    # Simulated annealing factor
    anneal = np.exp(-10*(1-rg))
    # Position update
    Positions = rg*donor + (1-rg)*Best_pos*anneal
    return Positions