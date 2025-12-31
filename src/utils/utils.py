import numpy as np

def compute_utility_regret(F_eval, F_all, beta_true):
    best_true_utility = float(np.max(F_all @ beta_true))

    if F_eval is None or len(F_eval) == 0:
        best_eval_utility = float('-inf')
    else:
        best_eval_utility = float(np.max(F_eval @ beta_true))

    return best_true_utility - best_eval_utility

def frechet_variance(X: np.ndarray) -> float:
    mu = X.mean(axis=0)  # shape (d,)
    sq_dists = np.sum((X - mu) ** 2, axis=1)  # shape (N,)
    # Frechet variance = average squared distance
    var_F = sq_dists.mean()
    return var_F