import pandas as pd
import numpy as np
from scipy.stats import norm
from itertools import combinations
from math import factorial
from functools import lru_cache
from itertools import combinations
from math import factorial


def shapley_values_discrete(X, Y, var_names=None, output_names=None, decimals=3):
    if var_names is None:
        var_names = [f"x{i+1}" for i in range(X.shape[1])]
    if output_names is None:
        output_names = [f"f{j+1}" for j in range(Y.shape[1])]

    d = X.shape[1]
    m = Y.shape[1]

    Xq = np.round(X, decimals=decimals)
    df_x = pd.DataFrame(Xq, columns=var_names)

    prob_tbl = df_x.value_counts(sort=False, normalize=True).reset_index()
    prob_tbl.columns = var_names + ["probability"]   # K×(d+1)

    df_y = pd.DataFrame(Y, columns=output_names)
    df_all = pd.concat([df_x, df_y], axis=1)
    g_means = df_all.groupby(var_names, sort=False)[output_names].mean().reset_index()

    U = prob_tbl.merge(g_means, on=var_names, how="left")
    p = U["probability"].to_numpy()
    G = U[output_names].to_numpy()

    E_g = (p[:, None] * G).sum(axis=0)
    Var_g = (p[:, None] * (G - E_g)**2).sum(axis=0)
    zero_var_mask = (Var_g <= 0)

    base_cols = var_names + ["probability"] + output_names

    Udf = U[base_cols].copy()

    @lru_cache(maxsize=None)
    def var_cond_all(subset_idx_tuple):
        if len(subset_idx_tuple) == 0:
            return np.zeros(m, dtype=float)

        cols = [var_names[i] for i in subset_idx_tuple]

        grp = Udf.groupby(cols, sort=False)
        P_S = grp["probability"].sum()
        PYG = grp.apply(lambda df: (df["probability"].to_numpy()[:,None] * df[output_names].to_numpy()).sum(axis=0))
        PYG_mat = np.vstack(PYG.values)
        mS = PYG_mat / P_S.to_numpy()[:, None]

        Var_vec = (P_S.to_numpy()[:,None] * (mS - E_g)**2).sum(axis=0)
        return Var_vec

    phi = np.zeros((d, m), dtype=float)
    d_fact = factorial(d)
    all_idx = tuple(range(d))

    for i in range(d):
        others = [j for j in all_idx if j != i]
        for k in range(0, d):
            for S in combinations(others, k):
                S = tuple(sorted(S))
                w = factorial(k) * factorial(d - k - 1) / d_fact
                v0 = var_cond_all(S)
                v1 = var_cond_all(tuple(sorted(S + (i,))))
                delta = v1 - v0
                nz = ~zero_var_mask
                if nz.any():
                    phi[i, nz] += w * (delta[nz] / Var_g[nz])

    for j in range(m):
        if not zero_var_mask[j]:
            s = phi[:, j].sum()
            if s > 0:
                phi[:, j] = phi[:, j] / s

    return pd.DataFrame(phi, index=var_names, columns=output_names)


def shapley_values_kernel(X, Y, var_names=None, output_names=None, bandwidth=0.1, leave_one_out=True):

    eps = 1e-12
    N, d = X.shape
    _, m = Y.shape

    if var_names is None:
        var_names = [f"x{i + 1}" for i in range(d)]
    if output_names is None:
        output_names = [f"f{j + 1}" for j in range(m)]

    # detect constant input variables
    var_inputs = np.var(X, axis=0)
    const_mask = var_inputs <= 1e-12

    # bandwidth per-dim
    if isinstance(bandwidth, (int, float)):
        h = np.full(d, float(bandwidth))
    else:
        h = np.asarray(bandwidth, dtype=float)
        assert h.shape == (d,), "bandwidth must be float or shape (d,) array"

    # precompute per-dimension N×N kernel weight matrices
    per_dim_weights = []
    for i in range(d):
        if const_mask[i]:
            per_dim_weights.append(None)  # skip constant
            continue
        xi = X[:, i][:, None]  # (N, 1)
        xj = X[:, i][None, :]  # (1, N)
        u = (xj - xi) / (h[i] + eps)
        Wi = np.exp(-0.5 * u ** 2)  # Gaussian kernel
        per_dim_weights.append(Wi)

    ones_N = np.ones((N, 1))

    # total variance Var(g_j)
    Var_g = np.var(Y, axis=0, ddof=1)  # (m,)
    zero_var_mask = Var_g <= 0

    @lru_cache(maxsize=None)
    def var_cond_kernel(subset_idx_tuple):
        # V(empty) = 0
        if len(subset_idx_tuple) == 0:
            return np.zeros(m, dtype=float)
        # definition-consistent fix: V(N) = Var(g)
        if len(subset_idx_tuple) == d:
            return Var_g.copy()

        # skip subsets that contain only constant vars, same as empty
        if all(const_mask[list(subset_idx_tuple)]):
            return np.zeros(m, dtype=float)

        # build kernel weight for subset
        W_S = None
        for idx in subset_idx_tuple:
            if const_mask[idx]:
                continue
            if W_S is None:
                W_S = per_dim_weights[idx].copy()
            else:
                W_S *= per_dim_weights[idx]

        if W_S is None:
            return np.zeros(m, dtype=float)

        if leave_one_out:
            np.fill_diagonal(W_S, 0.0)

        denom = W_S @ ones_N + eps  # (N,1)

        M_hat = (W_S @ Y) / denom
        Var_vec = M_hat.var(axis=0, ddof=1)
        return Var_vec

    # Shapley agg
    phi = np.zeros((d, m), dtype=float)
    d_fact = factorial(d)
    all_idx = tuple(range(d))

    for i in range(d):
        if const_mask[i]:
            continue  # constant vars → always 0
        others = [j for j in all_idx if j != i]
        for k in range(0, d):
            for S in combinations(others, k):
                S = tuple(sorted(S))
                S_i = tuple(sorted(S + (i,)))
                w = factorial(k) * factorial(d - k - 1) / d_fact

                v0 = var_cond_kernel(S)
                v1 = var_cond_kernel(S_i)
                delta = v1 - v0  # (m,)

                contrib = np.zeros(m, dtype=float)
                nz = ~zero_var_mask
                contrib[nz] = w * (delta[nz] / (Var_g[nz] + eps))
                phi[i] += contrib

    # force constant variables to zero
    phi[const_mask, :] = 0.0

    return pd.DataFrame(phi, index=var_names, columns=output_names)

