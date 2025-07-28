import numpy as np
import numpy as np
def heuristics_v2(node_attr, node_constraint):
    #EVOLVE-START
    demands = node_attr if node_attr.ndim == 1 else node_attr[:,0]
    n = len(demands)
    size_matrix = np.minimum(demands[:, None] + demands, node_constraint)
    ratio = np.minimum(demands[:, None] / demands, 1)  # element-wise ratio capped at 1
    feasible = (size_matrix <= node_constraint) & ~np.eye(n, dtype=bool)
    heuristic = feasible * ratio * np.log1p(demands[:, None] * demands)
    return np.nan_to_num(heuristic, nan=0)
    #EVOLVE-END
