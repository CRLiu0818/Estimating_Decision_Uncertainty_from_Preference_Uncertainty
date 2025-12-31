from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from .unified_problem import UnifiedProblem


class LP(UnifiedProblem):
    def __init__(self, r3: float = 14.0):
        super().__init__(objective_signs=np.array([+1.0, +1.0, -1.0, +1.0]), problem_type="LP") # min min max min
        self.r3 = float(r3)

        # variable bounds
        self._bounds: Dict[str, Dict[str, float | str]] = {
            "x1": {"type": "continuous", "lb": 4.5,  "ub": 23.0},
            "x2": {"type": "continuous", "lb": 6.0,  "ub": 20.0},
            "x3": {"type": "continuous", "lb": 1.0,  "ub": 5.0},
            "x4": {"type": "continuous", "lb": 1.0,  "ub": 3.0},
            "x5": {"type": "continuous", "lb": 0.0,  "ub": 15.0},
            "x6": {"type": "continuous", "lb": 34.0, "ub": 132.0},
            "x7": {"type": "continuous", "lb": 0.0,  "ub": 23.0},
            "x8": {"type": "continuous", "lb": 0.0,  "ub": 20.0},
        }

    def get_design_space(self) -> Dict[str, Any]:
        return self._bounds

    def _unpack(self, x: Dict[str, float]) -> Tuple[np.float64, ...]:
        x1 = np.float64(x["x1"])
        x2 = np.float64(x["x2"])
        x3 = np.float64(x["x3"])
        x4 = np.float64(x["x4"])
        x5 = np.float64(x["x5"])
        x6 = np.float64(x["x6"])
        th1 = np.float64(x["x7"])
        th2 = np.float64(x["x8"])
        return x1, x2, x3, x4, x5, x6, th1, th2

    def evaluate_objectives(self, x: Dict[str, float]) -> Dict[str, float]:
        x1, x2, x3, x4, x5, x6, th1, th2 = self._unpack(x)

        f_obj1 = max(x1 - x5, 0)
        f_obj2 = x6 + f_obj1 + max(x2 - x5, 0)
        f_obj3 = self.r3 * (x6 - 2.0*x5 - 3.414*x3 - 0.586*x4)
        f_obj4 = 2.0*self.r3 + (x6 - 2.0*x5)

        def clean_val(x, decimals=6, tol=1e-10):
            x = round(float(x), decimals)
            if abs(x) < tol:
                return 0.0
            return x

        return {
            "f1": clean_val(f_obj1),
            "f2": clean_val(f_obj2),
            "f3": clean_val(f_obj3),
            "f4": clean_val(f_obj4)
        }

    def is_feasible(self, x: Dict[str, float]) -> bool:
        eps = self.EPS
        x1, x2, x3, x4, x5, x6, th1, th2 = self._unpack(x)

        if (x3 - x4) > eps: # r4 <= r5
            return False
        if (3.414*x3 + 0.586*x4 + 2.0*x5 - x6) > eps:
            return False

        for name, meta in self._bounds.items():
            lb = meta["lb"]
            ub = meta["ub"]
            v = np.float64(x[name])
            if v < lb - eps or v > ub + eps:
                return False

        return True


    def _objective_linear_coeffs(self, idx: int) -> np.ndarray:
        r3 = self.r3
        c = np.zeros(8, dtype=np.float64)

        if idx == 0:
            # f1 = theta1
            c[6] = 1.0

        elif idx == 1:
            # f2 = f2 + theta1 + theta2
            c[5] = 1.0   # f2
            c[6] = 1.0   # theta1
            c[7] = 1.0   # theta2

        elif idx == 2:
            # f3 = r3 * (x6 - 2*x5 - 3.414*x3 - 0.586*x4)
            c[5] =  r3         # x6
            c[4] = -2.0 * r3   # x5
            c[2] = -3.414 * r3 # x3
            c[3] = -0.586 * r3 # x4

        elif idx == 3:
            # f4 = 2*r3 + (x6 - 2*x5)
            c[5] =  1.0  # x6
            c[4] = -2.0  # x5

        else:
            raise IndexError(f"Objective index {idx} out of range.")

        return c

    def _linear_constraints_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        A_ub = np.array([
            [ 1, 0, 0, 0, -1, 0, -1, 0],  # x1 - x5 - th1 <= 0
            [ 0, 1, 0, 0, -1, 0,  0,-1],  # x2 - f1 - th2 <= 0
            [ 0, 0, 1,-1,  0, 0,  0, 0],  # x3 - x4 <= 0
            [ 0, 0, 0, 0,  0, 0, -1, 0],  # -th1 <= 0
            [ 0, 0, 0, 0,  0, 0,  0,-1],  # -th2 <= 0
            [ 0, 0, 3.414, 0.586, 2,-1, 0, 0],  # 3.414*x3 + 0.586*x4 + 2*x5 - x6 <= 0
        ], dtype=np.float64)

        b_ub = np.zeros(6, dtype=np.float64)
        return A_ub, b_ub
