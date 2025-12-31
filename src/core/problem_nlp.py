# src/core/problem_nlp.py

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from .unified_problem import UnifiedProblem


class NLP(UnifiedProblem):
    def __init__(self, r3: float = 14.0):
        super().__init__(objective_signs=np.array([+1.0, +1.0, -1.0, +1.0]), problem_type="NLP") # min min max min
        self.r3 = float(r3)
        self.C1 = (16.0 + 2.0 * self.r3) ** 2  # 1936 when r3=14

        self._bounds: Dict[str, Dict[str, float | str]] = {
            "x1": {"type": "continuous", "lb": 4.5,  "ub": 23.0},   # r1
            "x2": {"type": "continuous", "lb": 6.0,  "ub": 20.0},   # r2
            "x3": {"type": "continuous", "lb": 1.0,  "ub": 5.0},    # r4
            "x4": {"type": "continuous", "lb": 1.0,  "ub": 3.0},    # r5
            "x5": {"type": "continuous", "lb": 0.0,  "ub": 15.0},   # f1
            "x6": {"type": "continuous", "lb": 34.0, "ub": 132.0},  # f2
            "x7": {"type": "continuous", "lb": 0.0,  "ub": 23.0},   # theta1
            "x8": {"type": "continuous", "lb": 0.0,  "ub": 20.0},   # theta2
        }

    def get_design_space(self) -> Dict[str, Any]:
        return self._bounds

    @property
    def variable_names(self) -> list[str]:
        names = list(self._bounds.keys())
        names.sort(key=lambda n: int(n[1:]))
        return names

    def _unpack(self, x: Dict[str, float]) -> Tuple[np.float64, ...]:
        x1 = np.float64(x["x1"])
        x2 = np.float64(x["x2"])
        x3 = np.float64(x["x3"])
        x4 = np.float64(x["x4"])
        x5 = np.float64(x["x5"])
        x6 = np.float64(x["x6"])
        x7 = np.float64(x["x7"])
        x8 = np.float64(x["x8"])
        return x1, x2, x3, x4, x5, x6, x7, x8

    def _f3_contact_term(self, x3, x4, x5, x6) -> float:
        return self.r3 * (x6 - 2.0*x5 - 3.414*x3 - 0.586*x4)

    def evaluate_objectives(self, x: Dict[str, float]) -> Dict[str, float]:
        x1, x2, x3, x4, x5, x6, x7, x8 = self._unpack(x)

        f1_obj = max(x1 - x5, 0)
        f2_obj = x6 + f1_obj + max(x2 - x5, 0)
        f3_obj = self._f3_contact_term(x3, x4, x5, x6)
        f4_obj = (f3_obj ** 2) + self.C1

        return {
            "f1": float(f1_obj),
            "f2": float(f2_obj),
            "f3": float(f3_obj),   # max
            "f4": float(f4_obj)
        }

    def is_feasible(self, x: Dict[str, float]) -> bool: # Need to check
        eps = self.EPS
        x1, x2, x3, x4, x5, x6, x7, x8 = self._unpack(x)

        # geometry
        if (x3 - x4) > eps:
            return False

        # contact constraint
        if self._f3_contact_term(x3, x4, x5, x6) < eps:
            return False

        # bounds
        for name, meta in self._bounds.items():
            lb = meta["lb"]; ub = meta["ub"]
            v = np.float64(x[name])
            if v < lb - eps or v > ub + eps:
                return False

        return True

