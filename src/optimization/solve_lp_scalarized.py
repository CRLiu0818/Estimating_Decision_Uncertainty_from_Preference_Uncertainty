from __future__ import annotations
import numpy as np
from typing import Dict, Any
from scipy.optimize import linprog
import gurobipy as gp
from gurobipy import GRB

def solve_beta_weighted_lp(problem, beta: np.ndarray, f_min: np.ndarray, f_max: np.ndarray, method: str = "highs") -> Dict[str, Any]:
    """
    Solve by linprog:  minimize_x   U(x; beta) = sum_j beta_j * f_j_norm(x)

    Parameters:
        problem : LP (subclass of UnifiedProblem)
        beta : np.ndarray shape (n_obj,) Preference weights (> 0)
        f_min, f_max : np.ndarray shape (n_obj,) Global min/max of the ORIGINAL objectives.

    Returns:
        dict with:
            "success": bool
            "status": str
            "x_opt": dict of x1...
            "f_raw": dict of f1... at x_opt (unnormalized)
            "f_norm": np.ndarray of normalized objectives at x_opt
            "U": float (scalarized utility)
            "beta"
    """
    beta = np.asarray(beta, dtype=np.float64)
    n_obj = problem.num_objectives

    if beta.shape[0] != n_obj:
        raise ValueError("beta length does not match number of objectives.")

    # variable bounds in canonical order
    space = problem.get_design_space()
    var_names = problem.variable_names
    lb = np.array([space[name]["lb"] for name in var_names], dtype=np.float64)
    ub = np.array([space[name]["ub"] for name in var_names], dtype=np.float64)
    bounds_lp = list(zip(lb, ub))

    # constraints
    A_ub, b_ub = problem._linear_constraints_matrix()

    n_vars = len(var_names)
    c_total = np.zeros(n_vars, dtype=np.float64)

    for j in range(n_obj):
        c_j = problem._objective_linear_coeffs(j)  # linear coeffs for raw f_j(x)

        f_j_min = f_min[j]
        f_j_max = f_max[j]
        delta_j = f_j_max - f_j_min
        if abs(delta_j) < 1e-12:
            delta_j = 1e-12  # avoid /0

        sign_j = problem.objective_signs[j]

        # minimize
        if sign_j == +1:
            coeff_norm_j = (1.0 / delta_j) * c_j

        # maximize
        else:
            coeff_norm_j = (-1.0 / delta_j) * c_j

        c_total += beta[j] * coeff_norm_j

    res = linprog(c=c_total, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method=method)

    if not res.success:
        return {
            "success": False,
            "status": res.message,
            "beta": beta,
        }

    x_sol_vec = res.x
    x_sol = {var_names[i]: float(x_sol_vec[i]) for i in range(n_vars)}

    f_raw_dict = problem.evaluate_objectives(x_sol)
    f_raw_vec = np.array([f_raw_dict[f"f{i+1}"] for i in range(n_obj)], dtype=np.float64)

    # normalized x
    f_norm_vec = np.empty_like(f_raw_vec, dtype=np.float64)
    for j in range(n_obj):
        sign_j = problem.objective_signs[j]
        f_j = f_raw_vec[j]
        f_j_min = f_min[j]
        f_j_max = f_max[j]
        delta_j = f_j_max - f_j_min
        if abs(delta_j) < 1e-12:
            delta_j = 1e-12

        if sign_j == +1:
            # minimize
            f_norm_vec[j] = (f_j - f_j_min) / delta_j
        else:
            # maximize
            f_norm_vec[j] = (f_j_max - f_j) / delta_j

    # U = beta^T f_norm
    U_val = float(np.dot(beta, f_norm_vec))

    return {
        "success": True,
        "status": "optimal",
        "beta": beta,
        "x_opt": x_sol, # dict of x1...
        "f_raw": f_raw_dict, # dict of {"f1":...,"f2":...}
        "f_norm": f_norm_vec,  # np.array([...])
        "U": U_val
    }


def solve_beta_weighted_lp_gurobi(problem, beta, f_min, f_max, tol_rc: float = 1e-12, tol_x: float = 1e-12, tiny_cost_rel: float = 1e-6):
    """
    Solve the beta-weighted LP with Gurobi by simplex and detect whether the optimal solution is unique.
    "multi_optima": returns which variables are involved in degeneracy.

    We split degenerate variables into:
        1. structural: variable never appears in ANY raw objective.
        2. geometric: caused by beta, variable has non-tiny effective cost in the scalarized objective but RC ~ 0 (true alternate optimal directions).
        3. numeric: variable has RC ~ 0 but its effective cost in U is numerically tiny (small betas). These are reported separately and not included in degenerate_vars.
    """
    beta = np.asarray(beta, dtype=np.float64)
    n_obj = problem.num_objectives

    if beta.shape[0] != n_obj:
        raise ValueError("beta length does not match number of objectives.")

    space = problem.get_design_space()
    var_names = problem.variable_names
    lb = np.array([space[name]["lb"] for name in var_names], dtype=np.float64)
    ub = np.array([space[name]["ub"] for name in var_names], dtype=np.float64)

    # Constraints: A_ub @ x <= b_ub
    A_ub, b_ub = problem._linear_constraints_matrix()

    n_vars = len(var_names)

    c_raw = np.zeros((n_obj, n_vars), dtype=np.float64)
    for j in range(n_obj):
        c_raw[j, :] = problem._objective_linear_coeffs(j)

    # Structural zero-cost variables: they never appear in ANY raw objective
    structural_zero_cost = set()
    for i in range(n_vars):
        if np.all(np.abs(c_raw[:, i]) <= 0.0):
            structural_zero_cost.add(var_names[i])

    c_total = np.zeros(n_vars, dtype=np.float64)

    for j in range(n_obj):
        c_j = c_raw[j, :]

        f_j_min = float(f_min[j])
        f_j_max = float(f_max[j])
        delta_j = f_j_max - f_j_min
        if abs(delta_j) < 1e-12:
            delta_j = 1e-12  # avoid division by zero

        sign_j = problem.objective_signs[j]

        if sign_j == +1:
            coeff_norm_j = (1.0 / delta_j) * c_j
        else:
            coeff_norm_j = (-1.0 / delta_j) * c_j

        c_total += beta[j] * coeff_norm_j

    if np.any(c_total != 0.0):
        max_abs_cost = float(np.max(np.abs(c_total)))
        tiny_cost_abs = tiny_cost_rel * max_abs_cost
    else:
        max_abs_cost = 0.0
        tiny_cost_abs = tol_rc

    model = gp.Model("beta_weighted_lp")
    model.Params.OutputFlag = 0
    # Simplex for VBasis and RC
    model.Params.Method = 0
    model.Params.Threads = 1

    # Decision variables x1..x8
    gvars = []
    for i, name in enumerate(var_names):
        gvar = model.addVar(
            lb=float(lb[i]),
            ub=float(ub[i]),
            vtype=GRB.CONTINUOUS,
            name=name,
        )
        gvars.append(gvar)

    model.update()

    # linear constraints A_ub x <= b_ub
    if A_ub is not None and A_ub.size > 0:
        m_ub = A_ub.shape[0]
        for k in range(m_ub):
            expr = gp.LinExpr()
            for i in range(n_vars):
                a_ki = float(A_ub[k, i])
                if abs(a_ki) > 0.0:
                    expr.add(gvars[i], a_ki)
            model.addLConstr(expr, GRB.LESS_EQUAL, float(b_ub[k]))

    obj_expr = gp.LinExpr()
    for i in range(n_vars):
        coeff = float(c_total[i])
        if coeff != 0.0:
            obj_expr.add(gvars[i], coeff)
    model.setObjective(obj_expr, GRB.MINIMIZE)

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        status_str = f"Gurobi status {model.Status}"
        return {
            "success": False,
            "status": status_str,
            "beta": beta,
        }

    x_sol_vec = np.array([v.X for v in gvars], dtype=np.float64)
    x_sol = {var_names[i]: float(x_sol_vec[i]) for i in range(n_vars)}

    f_raw_dict = problem.evaluate_objectives(x_sol)
    f_raw_vec = np.array(
        [f_raw_dict[f"f{i+1}"] for i in range(n_obj)],
        dtype=np.float64,
    )

    # Compute normalized objectives
    f_norm_vec = np.empty_like(f_raw_vec, dtype=np.float64)
    for j in range(n_obj):
        sign_j = problem.objective_signs[j]
        f_j = float(f_raw_vec[j])
        f_j_min = float(f_min[j])
        f_j_max = float(f_max[j])
        delta_j = f_j_max - f_j_min
        if abs(delta_j) < 1e-12:
            delta_j = 1e-12

        if sign_j == +1:
            # minimize
            f_norm_vec[j] = (f_j - f_j_min) / delta_j
        else:
            # maximize
            f_norm_vec[j] = (f_j_max - f_j) / delta_j

    U_val = float(np.dot(beta, f_norm_vec))

    vars_all = model.getVars()
    # Only check x1..x6, skip theta1/theta2
    real_indices = [0, 1, 2, 3, 4, 5]

    structural_deg = []  # structural zero-cost directions
    geometric_deg = []   # true alternate optima with non-tiny cost
    numeric_deg = []     # RC ~ 0 but effective cost tiny (beta)

    for idx in real_indices:
        v = vars_all[idx]
        name = var_names[idx]

        # Skip variables that are essentially fixed by their bounds
        if abs(v.UB - v.LB) <= tol_x:
            continue

        # Nonbasic variable with reduced cost ~ 0, candidate degenerate direction
        if v.VBasis != GRB.BASIC and abs(v.RC) <= tol_rc:
            # Structural case: variable never appears in ANY raw objective
            if name in structural_zero_cost:
                structural_deg.append(name)
            else:
                # Effective scalarized cost in U(x)
                eff_cost = abs(c_total[idx])
                if eff_cost <= tiny_cost_abs:
                    # Very small cost compared to the typical scale of c_total: this is most likely numeric degeneracy (tiny betas).
                    numeric_deg.append(name)
                else:
                    # Non-tiny cost but RC ~ 0: true geometric alternate optimum.
                    geometric_deg.append(name)

    multi_optima = (len(structural_deg) > 0) or (len(geometric_deg) > 0)

    degenerate_vars = structural_deg + geometric_deg

    return {
        "success": True,
        "status": "optimal",
        "beta": beta,
        "x_opt": x_sol, # dict of x1...
        "f_raw": f_raw_dict, # dict {"f1":..,"f2":..}
        "f_norm": f_norm_vec, # np.array([...])
        "U": U_val,
        "multi_optima": multi_optima,
        "degenerate_vars": degenerate_vars, # structural + geometric only
        "degenerate_structural": structural_deg,
        "degenerate_geometric": geometric_deg,
        "degenerate_numeric": numeric_deg,
        "c_total": c_total,
        "tiny_cost_abs": tiny_cost_abs,
        "structural_zero_cost": list(structural_zero_cost)
    }
