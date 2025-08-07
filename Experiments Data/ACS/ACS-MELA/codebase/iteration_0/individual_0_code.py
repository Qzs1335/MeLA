import numpy as np
def heuristics_v1(Positions, Best_pos, Best_score, rg):
    SearchAgents_no = Positions.shape[0]
    dim = Positions.shape[1]

    lb_array = np.zeros((SearchAgents_no, dim))
    ub_array = np.ones((SearchAgents_no, dim))

    rand_adjust = lb_array + (ub_array - lb_array) * np.random.rand(*Positions.shape)
    Positions = np.where((Positions < lb_array) | (Positions > ub_array), rand_adjust, Positions)

    p = np.arange(1, 361)
    cos_p = np.cos(np.deg2rad(p))
    r = np.random.rand(SearchAgents_no, 1) * rg
    R = 2 * rg * np.random.rand(SearchAgents_no, 1) - rg

    cp_indices = np.random.randint(0, SearchAgents_no, SearchAgents_no)

    is_exploit = (R >= -1) & (R <= 1)
    cumsum = np.cumsum(p / p.sum())
    teta_idxs = np.array([np.searchsorted(cumsum, np.random.rand()) for _ in range(SearchAgents_no)])

    delta = np.abs(np.random.rand(SearchAgents_no, dim) * Best_pos - Positions)
    cos_factor = np.array([cos_p[idx] for idx in teta_idxs]).reshape(-1, 1)

    exploit_mask = is_exploit.reshape(-1, 1)
    Positions = np.where(
        exploit_mask,
        Best_pos - r * delta * cos_factor,
        r * (Positions[cp_indices] - np.random.rand(SearchAgents_no, dim) * Positions)
    )

    return Positions