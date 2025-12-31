import numpy as np
import gurobipy as gp
from gurobipy import GRB

def _grad_constrs_wrt_z(r3: float):
    """
    Return gradients of constraints w.r.t. z and their type.
    Test feasibility direction only for ACTIVE constraints at z*.

    Each item: (name, grad, sense), sense: "eq" or "le" meaning grad^T d == 0 or <= 0
    """
    G = []

    # th1 >= x1 - x5  ->  (x1 - x5 - th1) <= 0
    g = np.zeros(8)
    g[0]=1
    g[4]=-1
    g[6]=-1
    G.append(("epi1", g, "le"))

    # th2 >= x2 - x5  ->  (x2 - f1 - th2) <= 0
    g = np.zeros(8)
    g[1]=1
    g[4]=-1
    g[7]=-1
    G.append(("epi2", g, "le"))

    # x4 >= x3 -> (x3 - x4) <= 0
    g = np.zeros(8)
    g[2]=1
    g[3]=-1
    G.append(("gear", g, "le"))

    # 3.414*x3 + 0.586*x4 + 2*x5 - x6 <= 0
    g = np.zeros(8)
    g[2]=3.414
    g[3]=0.586
    g[4]=2.0
    g[5]=-1.0
    G.append(("contact_le", g, "le"))

    return G


def grad_U_equiv(z: np.ndarray, beta: np.ndarray, r3: float, d1, d2, d3, d4):
    """
    Gradient of U w.r.t. z at z. z = [x1, x2, x3, x4, x5, x6, th1, th2]
    """
    b1, b2, b3, b4 = [float(x) for x in beta]

    # g3(z) = (x6 - 2 x5 - 3.414 x3 - 0.586 x4)*r3
    g3 = (z[5] - 2*z[4] - 3.414*z[2] - 0.586*z[3]) * r3
    dg3 = _grad_g3_wrt_z(r3)

    # f1_norm = (th1 - f1_min)/d1 -> grad w.r.t z only at th1
    grad = np.zeros(8)
    grad[6] += b1*(1.0/d1)

    # f2_norm = ((x6+th1+th2) - f2_min)/d2
    grad[5] += b2*(1.0/d2)
    grad[6] += b2*(1.0/d2)
    grad[7] += b2*(1.0/d2)

    # f3_norm = (f3_max - g3)/d3
    grad += b3 * (-(1.0/d3) * dg3)

    # f4_norm = (g4 - f4_min)/d4, and at optimum g4 = g3^2 + C1 => (b4/d4) * g3^2
    grad += b4 * (1.0/d4) * (2.0*g3) * dg3

    return grad



def _grad_g3_wrt_z(r3: float) -> np.ndarray:
    # z = [x1, x2, x3, x4, x5, x6, th1, th2]
    g = np.zeros(8, dtype=np.float64)
    g[2] = -3.414 * r3
    g[3] = -0.586 * r3
    g[4] = -2.0 * r3
    g[5] =  1.0 * r3
    return g


def hessian_U_equiv(beta: np.ndarray, r3: float, d4: float) -> np.ndarray:
    """
    Hessian of the equivalent eliminated objective term (beta4/d4)*g3^2, w.r.t z.
    This captures curvature induced by the active epigraph g4 >= g3^2 + C1.
    """
    beta4 = float(beta[3])
    grad = _grad_g3_wrt_z(r3)
    H = 2.0 * (beta4 / d4) * np.outer(grad, grad)
    return H



def solve_beta_weighted_nlp(problem, beta, f_min, f_max, timelimit=None, verbose=False):
    beta = np.asarray(beta, dtype=np.float64)
    r3   = problem.r3
    C1   = (16.0 + 2.0*r3)**2

    f1_min, f2_min, f3_min, f4_min = [float(v) for v in f_min]
    f1_max, f2_max, f3_max, f4_max = [float(v) for v in f_max]

    d1 = f1_max - f1_min
    d2 = f2_max - f2_min
    d3 = f3_max - f3_min
    d4 = f4_max - f4_min

    m = gp.Model("NLP")
    if not verbose:
        m.Params.OutputFlag = 0
    m.Params.FeasibilityTol = 1e-9
    m.Params.OptimalityTol  = 1e-9
    # m.Params.PoolSolutions = 10 # Check multi optima
    # m.Params.PoolSearchMode = 2 # Check multi optima
    # m.Params.PoolGap = 0
    m.Params.Threads = 1
    if timelimit is not None:
        m.Params.TimeLimit = timelimit

    bnds = problem.get_design_space()
    x1  = m.addVar(lb=bnds["x1"]["lb"], ub=bnds["x1"]["ub"], vtype=GRB.CONTINUOUS, name="x1")
    x2  = m.addVar(lb=bnds["x2"]["lb"], ub=bnds["x2"]["ub"], vtype=GRB.CONTINUOUS, name="x2")
    x3  = m.addVar(lb=bnds["x3"]["lb"], ub=bnds["x3"]["ub"], vtype=GRB.CONTINUOUS, name="x3")
    x4  = m.addVar(lb=bnds["x4"]["lb"], ub=bnds["x4"]["ub"], vtype=GRB.CONTINUOUS, name="x4")
    x5 = m.addVar(lb=bnds["x5"]["lb"], ub=bnds["x5"]["ub"], vtype=GRB.CONTINUOUS, name="x5")
    x6 = m.addVar(lb=bnds["x6"]["lb"], ub=bnds["x6"]["ub"], vtype=GRB.CONTINUOUS, name="x6")
    th1 = m.addVar(lb=bnds["x7"]["lb"], ub=bnds["x7"]["ub"], vtype=GRB.CONTINUOUS, name="theta1")
    th2 = m.addVar(lb=bnds["x8"]["lb"], ub=bnds["x8"]["ub"], vtype=GRB.CONTINUOUS, name="theta2")

    f2 = m.addVar(vtype=GRB.CONTINUOUS, name="f2")
    f3 = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="f3")
    f4 = m.addVar(vtype=GRB.CONTINUOUS, name="f4")

    # constraints
    m.addConstr(th1 >= x1 - x5)
    m.addConstr(th2 >= x2 - x5)
    m.addConstr(x4 >= x3)
    m.addConstr(f2 == x6 + th1 + th2)
    m.addConstr(f3 == (x6 - 2.0*x5 - 3.414*x3 - 0.586*x4)*r3)
    m.addConstr(f4 >= f3 * f3 + C1)
    m.addConstr(3.414*x3 + 0.586*x4 + 2.0*x5 - x6 <= 0.0)

    # Raw objectives
    f1_raw_expr = th1
    f2_raw_expr = f2
    f3_raw_expr = f3 # max
    f4_raw_expr = f4

    # Normalize objectives
    f1_norm_expr = (f1_raw_expr - f1_min) / d1
    f2_norm_expr = (f2_raw_expr - f2_min) / d2
    f3_norm_expr = (f3_max - f3_raw_expr) / d3  # max
    f4_norm_expr = (f4_raw_expr - f4_min) / d4

    U_expr = (beta[0]*f1_norm_expr + beta[1]*f2_norm_expr + beta[2]*f3_norm_expr + beta[3]*f4_norm_expr)

    m.setObjective(U_expr, GRB.MINIMIZE)

    # solve
    m.optimize()
    status = m.Status

    if status != GRB.OPTIMAL:
        return {
            "success": False,
            "status": f"Gurobi status {status}",
            "beta": beta,
            "f_norm": np.full(4, np.nan, dtype=np.float64),
        }

    theta1_val = float(th1.X)
    f2_val = float(f2.X)
    f3_val = float(f3.X)
    f4_val = float(f4.X)

    epi4_slack = float(f4_val - (f3_val*f3_val + C1))

    f1_norm_val = (theta1_val - f1_min)/d1
    f2_norm_val = (f2_val - f2_min)/d2
    f3_norm_val = (f3_max - f3_val)/d3
    f4_norm_val = (f4_val - f4_min)/d4

    f_norm_vec = np.array([f1_norm_val, f2_norm_val, f3_norm_val, f4_norm_val], dtype=np.float64)

    x_opt = {
        "x1": float(x1.X),
        "x2": float(x2.X),
        "x3": float(x3.X),
        "x4": float(x4.X),
        "x5": float(x5.X),
        "x6": float(x6.X),
        "x7": float(th1.X),
        "x8": float(th2.X),
    }

    f_raw_dict = problem.evaluate_objectives(x_opt)

    # Flat-direction diagnostics
    # tolerances for activity & directional feasibility
    tol_act   = 1e-7
    tol_slope = 1e-10
    tol_dir   = 1e-10

    # If epi4 is NOT binding, the equivalent eliminated curvature assumption is invalid.
    # In that case, skip.
    if epi4_slack > 1e-9:
        return {
            "success": True,
            "status": "optimal",
            "ObjVal": float(m.ObjVal),
            "beta": beta,
            "x_opt": x_opt,
            "f_raw": f_raw_dict,
            "f_norm": f_norm_vec,
            "U": float(np.dot(beta, f_norm_vec)),
            "sol_count": m.SolCount,
            "epi4_slack": epi4_slack,
            "note": "epi4 not binding; skip eliminated Hessian/nullspace flat-direction diagnostics.",
        }


    H = hessian_U_equiv(beta=beta, r3=r3, d4=d4)
    evals = np.linalg.eigvalsh(0.5*(H+H.T)) # numerical stability
    rank = int(np.sum(evals > 1e-10))

    eigvals, eigvecs = np.linalg.eigh(0.5*(H + H.T)) # numerical stability
    tol_null = 1e-10
    Null = eigvecs[:, eigvals <= tol_null]  # columns are directions d

    zstar = np.array([x_opt["x1"], x_opt["x2"], x_opt["x3"], x_opt["x4"], x_opt["x5"], x_opt["x6"], x_opt["x7"], x_opt["x8"]], dtype=float)

    gU = grad_U_equiv(z=zstar, beta=beta, r3=r3, d1=d1, d2=d2, d3=d3, d4=d4)

    # active constraints detection
    def _is_active(name, z):
        if name == "epi1": # x1 - x5 - th1 <= 0
            val = (z[0] - z[4] - z[6])
            return val >= -tol_act
        if name == "epi2": # x2 - x5 - th2 <= 0
            val = (z[1] - z[4] - z[7])
            return val >= -tol_act
        if name == "gear": # x3 - x4 <= 0
            val = (z[2] - z[3])
            return val >= -tol_act
        if name == "contact_le": # 3.414 x3 + 0.586 x4 + 2 x5 - x6 <= 0
            val = (3.414*z[2] + 0.586*z[3] + 2.0*z[4] - z[5])
            return val >= -tol_act
        return False

    G = _grad_constrs_wrt_z(r3)

    def bounds_feasible(d, z, lbs, ubs, tol_act, tol_dir):
        for i in range(len(z)):
            if abs(z[i] - lbs[i]) <= tol_act:
                if d[i] < -tol_dir:
                    return False
            if abs(z[i] - ubs[i]) <= tol_act:
                if d[i] > tol_dir:
                    return False
        return True

    slopes = []
    feasible_null_dirs = 0
    feasible_null_dirs_detail = []

    lbs = np.array([4.5, 6.0, 1.0, 1.0, 0.0, 34.0, 0.0, 0.0], dtype=float)
    ubs = np.array([23.0, 20.0, 5.0, 3.0, 15.0, 132.0, 23.0, 20.0], dtype=float)

    for i in range(Null.shape[1]):
        d = Null[:, i]
        slope = float(gU @ d)
        slopes.append(slope)

        # only check feasibility for first order flat directions
        if abs(slope) > tol_slope:
            continue

        ok = True

        # active inequality constraints: grad^T d <= 0
        for (nm, gradc, sense) in G:
            if not _is_active(nm, zstar):
                continue

            if sense == "le":
                if float(gradc @ d) > tol_dir:
                    ok = False
                    break
            elif sense == "eq":
                if abs(float(gradc @ d)) > tol_dir:
                    ok = False
                    break

        if not ok:
            continue

        # active bounds
        if not bounds_feasible(d, zstar, lbs, ubs, tol_act=tol_act, tol_dir=tol_dir):
            continue

        feasible_null_dirs += 1
        feasible_null_dirs_detail.append(i)

    var_names = ["x1", "x2", "x3", "x4", "x5", "x6", "th1", "th2"]
    flat_dirs = {}
    for k in feasible_null_dirs_detail:
        d = Null[:, k]
        d = d / (np.max(np.abs(d)) + 1e-15)
        flat_dirs[int(k)] = {var_names[i]: float(d[i]) for i in range(8)}

    return {
        "success": True,
        "status": "optimal",
        "ObjVal": float(m.ObjVal),
        "beta": beta,
        "x_opt": x_opt,
        "f_raw": f_raw_dict,
        "f_norm": f_norm_vec,
        "U": float(np.dot(beta, f_norm_vec)),
        "sol_count": m.SolCount,
        "epi4_slack": epi4_slack,
        "hessian_rank": rank,
        "nullspace_dim": int(Null.shape[1]),
        "nullspace_slopes": np.array(slopes),
        "num_flat_feasible_dirs": int(feasible_null_dirs),
        "flat_feasible_dir_indices": feasible_null_dirs_detail,
        "flat_dirs": flat_dirs,
    }

