from __future__ import annotations
import numpy as np
from numpy.random import Generator
from dataclasses import dataclass
from typing import Optional

@dataclass
class PreferenceModel:
    def sample(self, n: int, rng: Optional[Generator] = None) -> np.ndarray:
        raise NotImplementedError("Implement in subclass")

    def dim(self) -> int:
        raise NotImplementedError("Implement in subclass")

@dataclass
class DirichletPreference(PreferenceModel):
    alpha: np.ndarray  # shape (d_obj,), all > 0

    def sample(self, n: int, rng: Optional[Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.dirichlet(self.alpha, size=n)

    def dim(self) -> int:
        return self.alpha.shape[0]


@dataclass
class GaussianPreference(PreferenceModel):
    mean: np.ndarray        # shape (d_obj,)
    cov: np.ndarray         # shape (d_obj, d_obj)
    normalize: bool = False
    clip_nonnegative: bool = True

    def sample(self, n: int, rng: Optional[Generator] = None) -> np.ndarray:

        if rng is None:
            rng = np.random.default_rng()

        raw = rng.multivariate_normal(mean=self.mean, cov=self.cov, size=n)

        if self.clip_nonnegative:
            raw = np.maximum(raw, 1e-5)

        if self.normalize:
            sums = raw.sum(axis=1, keepdims=True)
            # avoid divide-by-zero: if sum==0, fallback to uniform
            zero_mask = (sums.squeeze(-1) == 0)
            if np.any(zero_mask):
                d = raw.shape[1]
                raw[zero_mask] = 1.0 / d
                sums[zero_mask] = 1.0
            beta = raw / sums
        else:
            beta = raw

        return beta

    def dim(self) -> int:
        return self.mean.shape[0]