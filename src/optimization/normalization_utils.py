from __future__ import annotations
from gurobipy import GRB
from typing import Tuple
from scipy.optimize import linprog
import numpy as np
import gurobipy as gp


def compute_obj_bounds_lp(problem, method: str = "highs") -> Tuple[np.ndarray, np.ndarray]:
    n_obj = problem.num_objectives

    space = problem.get_design_space()
    lb = np.array([space[name]["lb"] for name in problem.variable_names], dtype=np.float64)
    ub = np.array([space[name]["ub"] for name in problem.variable_names], dtype=np.float64)
    bounds_lp = list(zip(lb, ub))

    A_ub, b_ub = problem._linear_constraints_matrix()

    f_min_list = []
    f_max_list = []

    for i in range(n_obj):
        c = problem._objective_linear_coeffs(i)

        # minimize f
        res_min = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method=method)
        if not res_min.success:
            raise RuntimeError(f"Min f{i+1} failed: {res_min.message}")
        f_i_min = res_min.fun

        # maximize f (minimize -f)
        res_max = linprog(c=-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method=method)
        if not res_max.success:
            raise RuntimeError(f"Max f{i+1} failed: {res_max.message}")
        f_i_max = -res_max.fun

        if i == 3:
            const_term = 2.0 * problem.r3
            f_i_min += const_term
            f_i_max += const_term

        f_min_list.append(f_i_min)
        f_max_list.append(f_i_max)

    f_min = np.array(f_min_list, dtype=np.float64)
    f_max = np.array(f_max_list, dtype=np.float64)

    return f_min, f_max


def compute_obj_bounds_nlp(problem, verbose: bool = False, timelimit: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
    r3 = problem.r3
    C1 = (16.0 + 2.0 * r3) ** 2
    n_obj = problem.num_objectives  # 4

    def build_model():
        m = gp.Model("bounds_nlp")
        if not verbose:
            m.Params.OutputFlag = 0
        m.Params.NonConvex = 2
        m.Params.FeasibilityTol = 1e-9
        m.Params.OptimalityTol = 1e-9
        if timelimit is not None:
            m.Params.TimeLimit = timelimit

        # Decision variables with bounds from problem.get_design_space()
        space = problem.get_design_space()
        x1 = m.addVar(lb=space["x1"]["lb"], ub=space["x1"]["ub"], vtype=GRB.CONTINUOUS, name="x1")
        x2 = m.addVar(lb=space["x2"]["lb"], ub=space["x2"]["ub"], vtype=GRB.CONTINUOUS, name="x2")
        x3 = m.addVar(lb=space["x3"]["lb"], ub=space["x3"]["ub"], vtype=GRB.CONTINUOUS, name="x3")
        x4 = m.addVar(lb=space["x4"]["lb"], ub=space["x4"]["ub"], vtype=GRB.CONTINUOUS, name="x4")
        x5 = m.addVar(lb=space["x5"]["lb"], ub=space["x5"]["ub"], vtype=GRB.CONTINUOUS, name="x5")
        x6 = m.addVar(lb=space["x6"]["lb"], ub=space["x6"]["ub"], vtype=GRB.CONTINUOUS, name="x6")
        th1 = m.addVar(lb=space["x7"]["lb"], ub=space["x7"]["ub"], vtype=GRB.CONTINUOUS, name="theta1")
        th2 = m.addVar(lb=space["x8"]["lb"], ub=space["x8"]["ub"], vtype=GRB.CONTINUOUS, name="theta2")

        f2 = m.addVar(vtype=GRB.CONTINUOUS, name="f2")
        f3 = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="f3")
        f4 = m.addVar(vtype=GRB.CONTINUOUS, name="f4")

        # Constraints
        # theta1 >= x1 - x5
        m.addConstr(th1 >= x1 - x5)
        # theta2 >= x2 - x5
        m.addConstr(th2 >= x2 - x5)
        # theta1, theta2 >= 0  (bounded by lb=0)
        m.addConstr(th1 >= 0.0)
        m.addConstr(th2 >= 0.0)
        # x4 >= x3
        m.addConstr(x4 >= x3)
        # linear constraint:
        m.addConstr(3.414 * x3 + 0.586 * x4 + 2.0 * x5 - x6 <= 0.0)

        # define f2, f3, f4
        m.addConstr(f2 == x6 + th1 + th2)
        m.addConstr(f3 == (x6 - 2.0 * x5 - 3.414 * x3 - 0.586 * x4) * r3)
        # quadratic
        m.addConstr(f4 == f3 * f3 + C1)

        f1_expr = th1
        f2_expr = f2
        f3_expr = f3 # should be maximized
        f4_expr = f4 # quadratic

        f_exprs = [f1_expr, f2_expr, f3_expr, f4_expr]

        return m, f_exprs

    f_min_list = []
    f_max_list = []

    for i in range(n_obj):
        # minimize fi
        m_min, f_exprs_min = build_model()
        m_min.setObjective(f_exprs_min[i], GRB.MINIMIZE)
        m_min.optimize()

        if m_min.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Min f{i+1} not OPTIMAL, status={m_min.Status}")

        f_i_min = float(f_exprs_min[i].X)

        # maximize f  (minimize -f)
        m_max, f_exprs_max = build_model()
        m_max.setObjective(f_exprs_max[i], GRB.MAXIMIZE)
        m_max.optimize()

        if m_max.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Max f{i+1} not OPTIMAL, status={m_max.Status}")

        f_i_max = float(f_exprs_max[i].X)

        f_min_list.append(f_i_min)
        f_max_list.append(f_i_max)

    f_min = np.array(f_min_list, dtype=np.float64)
    f_max = np.array(f_max_list, dtype=np.float64)

    return f_min, f_max


def normalize_objectives(f_values: np.ndarray, f_min: np.ndarray, f_max: np.ndarray) -> np.ndarray:
    f_values = np.array(f_values, dtype=np.float64)
    f_min = np.array(f_min, dtype=np.float64)
    f_max = np.array(f_max, dtype=np.float64)

    delta = f_max - f_min
    delta[np.abs(delta) < 1e-12] = 1e-12

    f_norm = (f_values - f_min) / delta
    return f_norm


def extract_and_normalize_X(results, bounds):
    var_names = list(bounds.keys())
    d = len(var_names)

    X_raw = np.zeros((len(results), d))
    for i, r in enumerate(results):
        x_dict = r["x_opt"]
        X_raw[i, :] = [x_dict[v] for v in var_names]

    X_norm = np.zeros_like(X_raw)
    for j, v in enumerate(var_names):
        lb, ub = bounds[v]["lb"], bounds[v]["ub"]
        X_norm[:, j] = (X_raw[:, j] - lb) / (ub - lb)
    return X_raw, X_norm
