import numpy as np

def sturges_bins(X):
    N, d = X.shape
    k = int(np.ceil(np.log2(N))) + 1
    return [k for _ in range(d)]

def freedman_diaconis_bins(X):
    N, d = X.shape
    n_bins_list = []
    for i in range(d):
        data = X[:, i]
        iqr = np.subtract(*np.percentile(data, [75, 25]))
        bin_width = 2 * iqr / (N ** (1 / 3))
        if bin_width > 0:
            n_bins = int(np.ceil((data.max() - data.min()) / bin_width))
        else:
            n_bins = 1  # fallback if IQR = 0
        n_bins_list.append(max(n_bins, 1))
    return n_bins_list

def silverman_bandwidths(X):
    N, d = X.shape
    stds = np.std(X, axis=0, ddof=1)
    h = 1.06 * stds * N ** (-1 / 5)
    return h