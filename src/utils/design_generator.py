import numpy as np
from scipy.stats import qmc

def generate_hypercube_samples(bounds: dict, n_samples: int, exclude_vars: list) -> np.ndarray:
    """
    bounds : dict
        Dictionary with variable metadata, e.g. {'x1': {'type': 'continuous', 'lb': 4.5, 'ub': 23.0}, ...}
    n_samples : int
        Number of samples to generate.
    exclude_vars : list of str, optional
        Variables to exclude from sampling (e.g. linearization vars).
    """
    active_vars = [k for k in sorted(bounds.keys()) if k not in exclude_vars]
    n_vars = len(active_vars)

    lb = np.array([bounds[k]["lb"] for k in active_vars], dtype=float)
    ub = np.array([bounds[k]["ub"] for k in active_vars], dtype=float)

    # LHS in [0,1]^n
    sampler = qmc.LatinHypercube(d=n_vars, seed=42)
    unit_samples = sampler.random(n=n_samples)

    # Scale samples to the specified bounds
    scaled_samples = qmc.scale(unit_samples, lb, ub)

    return scaled_samples, active_vars


def add_linearization_vars(samples: np.ndarray, var_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Given LHS samples to derive theta1 (x7) and theta2 (x8)
    """
    # Extract indices for r1 (x1), r2 (x2), f1 (x5)
    i_x1 = var_names.index("x1")
    i_x2 = var_names.index("x2")
    i_x5 = var_names.index("x5")

    theta1 = np.maximum(0.0, samples[:, i_x1] - samples[:, i_x5])
    theta2 = np.maximum(0.0, samples[:, i_x2] - samples[:, i_x5])

    extended = np.column_stack([samples, theta1, theta2])
    new_vars = var_names + ["x7", "x8"]
    return extended, new_vars
