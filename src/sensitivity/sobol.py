import pandas as pd
import numpy as np
from scipy.stats import norm
from itertools import combinations
from math import factorial
from functools import lru_cache

def sobol_first_order_discrete(X, Y, var_names=None, output_names=None, decimals=3):
    if var_names is None:
        var_names = [f"t{i+1}" for i in range(X.shape[1])]
    if output_names is None:
        output_names = [f"g{j+1}" for j in range(Y.shape[1])]

    Xq = np.round(X, decimals=decimals)
    df_x = pd.DataFrame(Xq, columns=var_names)

    value_counts = df_x.value_counts(normalize=True).reset_index()
    value_counts.columns = var_names + ['probability']

    t_values = value_counts[var_names].values
    prob = value_counts['probability'].values
    K, d = t_values.shape
    m = Y.shape[1]

    g_values = np.zeros((K, m))
    for k, row in value_counts[var_names].iterrows():
        mask = (df_x == row.values).all(axis=1)
        g_values[k] = Y[mask].mean(axis=0)

    S1 = np.zeros((d, m))
    for j in range(m):
        g = g_values[:, j]
        E_g = np.sum(prob * g)
        Var_g = np.sum(prob * (g - E_g) ** 2)

        for i in range(d):
            levels = np.unique(t_values[:, i])
            E_g_given, P_given = [], []
            for v in levels:
                idx = (t_values[:, i] == v)
                p_v = np.sum(prob[idx])
                if p_v > 0:
                    E_v = np.sum(prob[idx] * g[idx]) / p_v
                    E_g_given.append(E_v)
                    P_given.append(p_v)
            E_g_given = np.array(E_g_given); P_given = np.array(P_given)
            Var_E = np.sum(P_given * (E_g_given - E_g) ** 2)
            S1[i, j] = Var_E / Var_g if Var_g > 0 else 0.0

    return pd.DataFrame(S1, index=var_names, columns=output_names)


def sobol_total_order_discrete(X, Y, var_names=None, output_names=None, decimals=3):
    if var_names is None:
        var_names = [f"x{i+1}" for i in range(X.shape[1])]
    if output_names is None:
        output_names = [f"f{j+1}" for j in range(Y.shape[1])]

    Xq = np.round(X, decimals=decimals)
    df = pd.DataFrame(Xq, columns=var_names)
    grouped = df.value_counts(sort=False, normalize=True).reset_index()
    grouped.columns = var_names + ["probability"]

    df_all = df.copy()
    f_df = pd.DataFrame(Y, columns=output_names)
    df_all = pd.concat([df_all, f_df], axis=1)
    g_means = df_all.groupby(var_names, sort=False)[output_names].mean().reset_index()

    uniq = grouped.merge(g_means, on=var_names, how="left")
    prob = uniq["probability"].to_numpy()                   # (K,)
    g_all = uniq[output_names].to_numpy()                   # (K, m)

    m = g_all.shape[1]
    d = len(var_names)
    ST = np.zeros((d, m), dtype=float)

    E_g = (prob[:, None] * g_all).sum(axis=0)               # (m,)
    Var_g = (prob[:, None] * (g_all - E_g)**2).sum(axis=0)  # (m,)

    for i, vname in enumerate(var_names):
        other = [vn for vn in var_names if vn != vname]
        sub = uniq.groupby(other, sort=False)

        for j in range(m):
            cond_var_acc = 0.0
            for _, block in sub:
                p = block["probability"].to_numpy()
                if p.sum() <= 0:
                    continue
                g = block[output_names[j]].to_numpy()
                w = p / p.sum() # P(X_i | X_-i)
                mean_g = np.sum(w * g)
                var_g = np.sum(w * (g - mean_g)**2) if g.size > 1 else 0.0
                cond_var_acc += p.sum() * var_g

            if Var_g[j] > 0:
                ST[i, j] = cond_var_acc / Var_g[j]
            else:
                ST[i, j] = 0.0

    return pd.DataFrame(ST, index=var_names, columns=output_names)


def sobol_first_order_continuous(X, Y, var_names=None, output_names=None, n_bins=30, clip=True, verbose=False):
    N, d = X.shape
    _, m = Y.shape
    S_first = np.zeros((d, m))

    for j in range(m):  # g_j
        y = Y[:, j]
        var_y = np.var(y, ddof=1)

        for i in range(d):  # t_i
            bins = np.linspace(0, 1, n_bins + 1)
            digitized = np.digitize(X[:, i], bins) - 1
            cond_means = np.zeros(n_bins)
            bin_counts = np.zeros(n_bins)

            for b in range(n_bins):
                idx = (digitized == b)
                if np.any(idx):
                    cond_means[b] = np.mean(y[idx])
                    bin_counts[b] = np.sum(idx)

            valid_bins = bin_counts > 0
            cond_means_valid = cond_means[valid_bins]
            weights = bin_counts[valid_bins] / np.sum(bin_counts[valid_bins])

            global_mean = np.mean(y)
            var_cond_mean = np.sum(weights * (cond_means_valid - global_mean) ** 2)

            S_ij = var_cond_mean / var_y
            if S_ij > 1.0 and clip:
                if verbose:
                    print(f"S_ij clipped: t={i} ({var_names[i] if var_names else 't'+str(i+1)}), "
                          f"g={j} ({output_names[j] if output_names else 'g'+str(j+1)}), S_ij={S_ij:.4f} > 1")
                S_ij = 1.0

            S_first[i, j] = S_ij

    if var_names is None:
        var_names = [f"x{i+1}" for i in range(d)]
    if output_names is None:
        output_names = [f"f{j+1}" for j in range(m)]

    df_sobol = pd.DataFrame(S_first, index=var_names, columns=output_names)
    return df_sobol


def sobol_total_order_continuous(X, Y, var_names=None, output_names=None, n_bins=30, clip=True, verbose=True):
    N, d = X.shape
    _, m = Y.shape
    S_total = np.zeros((d, m))

    for j in range(m):  # f
        y = Y[:, j]
        var_y = np.var(y, ddof=1)

        for i in range(d):  # x
            t_minus_i = np.delete(X, i, axis=1)

            bin_edges = [np.linspace(0, 1, n_bins + 1) for _ in range(d - 1)]
            digitized = np.stack([
                np.digitize(t_minus_i[:, k], bin_edges[k]) - 1
                for k in range(d - 1)
            ], axis=1)

            bin_keys = [tuple(row) for row in digitized]
            bin_dict = {}

            for idx, key in enumerate(bin_keys):
                bin_dict.setdefault(key, []).append(idx)

            cond_vars = []
            weights = []

            for key, idx_list in bin_dict.items():
                if len(idx_list) < 2:
                    continue
                y_subset = y[idx_list]
                cond_vars.append(np.var(y_subset, ddof=1))
                weights.append(len(idx_list))

            if len(weights) == 0:
                S_Tij = 0.0
            else:
                weights = np.array(weights) / np.sum(weights)
                expected_cond_var = np.sum(weights * np.array(cond_vars))
                S_Tij = expected_cond_var / var_y
                if S_Tij > 1.0 and clip:
                    if verbose:
                        print(f"S_Tij clipped: t={i} ({var_names[i] if var_names else 't'+str(i+1)}), "
                              f"g={j} ({output_names[j] if output_names else 'g'+str(j+1)}), S_Tij={S_Tij:.4f} > 1")
                    S_Tij = 1.0

            S_total[i, j] = S_Tij

    if var_names is None:
        var_names = [f"x{i+1}" for i in range(d)]
    if output_names is None:
        output_names = [f"f{j+1}" for j in range(m)]

    df_total = pd.DataFrame(S_total, index=var_names, columns=output_names)
    return df_total


def sobol_first_order_kernel(X, Y, var_names=None, output_names=None, bandwidth=0.1, clip=True, verbose=True, eps_bw=1e-6):
    N, d = X.shape
    _, m = Y.shape
    S_first = np.zeros((d, m))

    # prepare bandwidths
    if isinstance(bandwidth, (float, int)):
        bandwidths = np.full(d, bandwidth)
    else:
        bandwidths = np.asarray(bandwidth)
        assert bandwidths.shape[0] == d, "bandwidth must be float or array of shape (d,)"

    for j in range(m):  # loop over outputs
        y = Y[:, j]
        var_y = np.var(y, ddof=1)

        for i in range(d):  # loop over inputs
            xi = X[:, i]  # (N,)
            # h_i = bandwidths[i]
            h_i = np.maximum(bandwidths[i], eps_bw)

            y_hat = np.zeros(N)
            for k in range(N):
                u = (xi - xi[k]) / h_i
                weights = norm.pdf(u)
                weights_sum = np.sum(weights)
                if weights_sum > 0:
                    y_hat[k] = np.sum(weights * y) / weights_sum
                else:
                    y_hat[k] = 0.0

            var_cond_mean = np.var(y_hat, ddof=1)
            S_ij = var_cond_mean / var_y

            if S_ij > 1.0 and clip:
                if verbose:
                    print(f"[Kernel] S_ij clipped: t={i} ({var_names[i] if var_names else 't' + str(i + 1)}), "
                          f"g={j} ({output_names[j] if output_names else 'g' + str(j + 1)}), S_ij={S_ij:.4f} > 1")
                S_ij = 1.0

            S_first[i, j] = S_ij

    if var_names is None:
        var_names = [f"x{i + 1}" for i in range(d)]
    if output_names is None:
        output_names = [f"f{j + 1}" for j in range(m)]

    return pd.DataFrame(S_first, index=var_names, columns=output_names)


def sobol_total_order_kernel(X, Y, var_names=None, output_names=None, bandwidth=0.1, clip=True, verbose=True, leave_one_out=False, eps=1e-12, eps_bw=1e-6):

    N, d = X.shape
    _, m = Y.shape
    S_total = np.zeros((d, m))

    # per-dim bandwidths
    if isinstance(bandwidth, (float, int)):
        h_full = np.full(d, float(bandwidth))
    else:
        h_full = np.asarray(bandwidth, dtype=float)
        assert h_full.shape[0] == d, "bandwidth must be float or array of shape (d,)"
    h_full = np.maximum(h_full, eps_bw)

    if var_names is None:
        var_names = [f"x{i+1}" for i in range(d)]
    if output_names is None:
        output_names = [f"f{j+1}" for j in range(m)]

    # variance of outputs
    var_y_all = np.var(Y, axis=0, ddof=1)
    zero_var_mask = var_y_all <= 0

    # detect constant inputs
    stds = np.std(X, axis=0, ddof=1)
    active_mask_global = stds > eps_bw

    for j in range(m):
        y = Y[:, j]
        var_y = var_y_all[j]
        if var_y <= 0:
            continue

        for i in range(d):
            # constant => 0
            if not active_mask_global[i]:
                S_total[i, j] = 0.0
                continue

            keep = active_mask_global.copy()
            keep[i] = False
            idx_keep = np.where(keep)[0]

            if idx_keep.size == 0:
                # only this variable is active, then ST = 1
                S_total[i, j] = 1.0
                continue

            Xmi = X[:, idx_keep]
            h_i = np.maximum(h_full[idx_keep], eps_bw)

            cond_vars = np.zeros(N)
            for k in range(N):
                x0 = Xmi[k]
                diffs = (Xmi - x0) / h_i
                w = norm.pdf(diffs)
                weights = np.prod(w, axis=1)

                if leave_one_out:
                    weights[k] = 0.0

                w_sum = weights.sum()
                if w_sum <= eps:
                    cond_vars[k] = np.var(y, ddof=1)
                else:
                    mean_y = np.sum(weights * y) / w_sum
                    var_y_cond = np.sum(weights * (y - mean_y) ** 2) / w_sum
                    cond_vars[k] = var_y_cond

            expected_cond_var = float(np.mean(cond_vars))
            S_Tij = expected_cond_var / (var_y + eps)

            if S_Tij > 1.0 and clip:
                if verbose:
                    print(f"[Kernel] S_T clipped: {var_names[i]}–{output_names[j]} -> {S_Tij:.4f} > 1")
                S_Tij = 1.0

            S_total[i, j] = S_Tij

    return pd.DataFrame(S_total, index=var_names, columns=output_names)
