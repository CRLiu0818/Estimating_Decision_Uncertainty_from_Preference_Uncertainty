from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class UnifiedProblem(ABC):
    """
    Abstract base class for a multi-objective engineering design problem.
    - Define decision variables with bounds
    - Define objectives
    - Define feasibility constraints
    - Provide normalization hooks (but NOT compute min/max itself)
    """

    EPS = 1e-8  # numerical tolerance for feasibility checking

    def __init__(self, objective_signs: np.ndarray, problem_type: str):
        """
        objective_signs[i] = +1 → fi is a "minimize" objective
        objective_signs[i] = -1 → fi is a "maximize" objective
        """
        self.objective_signs = np.array(objective_signs, dtype=float)
        self.problem_type: str = problem_type

        # Will be filled later by optimization layer
        self._f_min_global: np.ndarray | None = None
        self._f_max_global: np.ndarray | None = None

    # Design space
    @abstractmethod
    def get_design_space(self) -> Dict[str, Any]:
        """
        Return dict describing x1, x2, ..., xN.
        Each entry:
            { "type": "continuous", "lb": float, "ub": float }
        Variable names MUST be "x1","x2",...,"xN".
        """
        raise NotImplementedError

    @property
    def variable_names(self) -> list[str]:
        """
        Canonical variable order used by solvers.
        ["x1","x2","x3",...]
        """
        names = list(self.get_design_space().keys())
        names.sort(key=lambda n: int(n[1:]))  # sort by numeric suffix
        return names

    # Objectives
    @abstractmethod
    def evaluate_objectives(self, x: Dict[str, float]) -> Dict[str, float]:
        """
        Return:
            { "f1": ..., "f2": ...}
        Keys MUST be exactly "f1","f2",... in order.
        """
        raise NotImplementedError

    @property
    def num_objectives(self) -> int:
        # Number of objectives M
        return len(self.objective_signs)

    def objective_vector(self, x: Dict[str, float]) -> np.ndarray:
        # Return [f1(x), f2(x), ...] as np.ndarray.
        fvals = self.evaluate_objectives(x)
        vals = [fvals[f"f{i+1}"] for i in range(self.num_objectives)]
        return np.array(vals, dtype=np.float64)

    def signed_objectives(self, x: Dict[str, float]) -> np.ndarray:
        return self.objective_vector(x) * self.objective_signs

    # Feasibility
    @abstractmethod
    def is_feasible(self, x: Dict[str, float]) -> bool:
        # Check problem constraints and bounds with tolerance EPS.
        raise NotImplementedError

    def evaluate_constraints(self, x: Dict[str, float]) -> np.ndarray | None:
        # Optional: return g(x) s.t. g_k(x) <= 0 means feasible. Subclasses can override.
        return None

    # Normalization hooks (filled by optimization layer)
    def set_global_objective_bounds(self, f_min: np.ndarray, f_max: np.ndarray) -> None:
        # Store global min/max for each objective fi
        # (computed externally by the optimization layer).
        self._f_min_global = np.array(f_min, dtype=np.float64)
        self._f_max_global = np.array(f_max, dtype=np.float64)

    def normalize_objectives(self, x: Dict[str, float]) -> np.ndarray:
        """
        Normalize objectives to [0,1] and align "lower = better".
        For each i:
          if sign[i] = +1 (minimize fi):
              norm_i = (fi(x) - f_min[i]) / (f_max[i] - f_min[i])
          else (maximize fi):
              norm_i = (f_max[i] - fi(x)) / (f_max[i] - f_min[i])
        """
        if self._f_min_global is None or self._f_max_global is None:
            raise RuntimeError(
                "Global objective bounds not set. "
                "Call set_global_objective_bounds(...) first."
            )

        f_vec = self.objective_vector(x)  # [f1,f2,...]
        f_min = self._f_min_global
        f_max = self._f_max_global

        delta = f_max - f_min
        delta[np.abs(delta) < 1e-12] = 1e-12  # avoid /0

        norm = np.empty_like(f_vec, dtype=np.float64)
        for i in range(self.num_objectives):
            if self.objective_signs[i] == +1:  # minimize
                norm[i] = (f_vec[i] - f_min[i]) / delta[i]
            else:  # maximize
                norm[i] = (f_max[i] - f_vec[i]) / delta[i]

        return norm
