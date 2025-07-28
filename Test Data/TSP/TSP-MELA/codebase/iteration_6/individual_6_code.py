import numpy as np
import numpy as np  

def heuristics_v2(distance_matrix):  
    #EVOLVE-START  
    alpha = 1.7                       
    beta = 1.3                      
    adaptive_eps = max(1e-16, np.min(distance_matrix[distance_matrix > 0])/1e6)  
    
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)  
    mask = distance_matrix <= 0  
    distance_matrix[mask] = np.inf  
    np.fill_diagonal(distance_matrix, np.inf)  

    visibility = np.exp(-beta * np.log(distance_matrix + adaptive_eps))  
    pheromone = np.exp(-0.2 * np.arange(len(distance_matrix))[:,None])  
    heuristic = np.exp(alpha * np.log(pheromone + adaptive_eps)) * visibility  

    softmax = np.exp(heuristic - np.max(heuristic)) / np.sum(np.exp(heuristic))  
    return softmax / softmax.sum() if softmax.sum() > 0 else softmax 
    #EVOLVE-END