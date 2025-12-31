import numpy as np

def project_positive_weights(mu, eps=1e-5):
    mu = np.asarray(mu, float)
    mu = np.clip(mu, eps, None)
    return mu / mu.sum()
