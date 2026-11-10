"""Pure-torch ridge solvers for the drift map W (d, d) of Stage 2, shared by
main.fit_drift_map (optional train_drift_solver='woodbury' and the
train_drift_lambda_rule='gcv' selection), bench_drift_solve.py (E3) and the
E9 oracle map. No timm / peft import; CPU float64 like production.

The identity-anchored ridge (train_drift_ridge='identity', the paper's map)
solves, with X = F_old (n, d) and Y = F_new - F_old,

    Delta = argmin ||Y - X Delta^T||_F^2 + lam ||Delta||_F^2,   W = I + Delta,

and the zero-anchored ridge uses Y = F_new and W = Delta. ridge_*_dense are
verbatim transcriptions of main.py:1205-1212 (normal equations, a (d, d)
solve with d right-hand sides); ridge_*_woodbury use the push-through
identity X (X^T X + lam I_d)^-1 = (X X^T + lam I_n)^-1 X, an (n, n) solve,
cheaper when n < d. Both return W in float64; main.py casts to float32 after
its log line.
"""
import torch

# The covariance-transport kernels live in covstats (one owner); they are
# re-exported here because bench_drift_solve.py (E3) and the tier-b test reach
# the whole drift-solve toolkit through this module. covstats imports only torch.
from covstats import transport_diag, transport_full  # noqa: F401

#: Default log grid of the GCV selection (train_drift_lambda_rule='gcv').
GCV_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


# ---------------------------------------------------------------------------
# Dense solvers (main.py:1205-1212, verbatim arithmetic on float64 inputs)
# ---------------------------------------------------------------------------

def ridge_identity_dense(F_old, F_new, lam):
    """W = I + Delta with A = F_old^T F_old + lam I, Cd = (F_new - F_old)^T F_old,
    Delta = solve(A, Cd^T)^T. Verbatim from fit_drift_map (identity branch)."""
    F_old, F_new = F_old.double(), F_new.double()
    d = F_old.size(1)
    A = F_old.T @ F_old + lam * torch.eye(d, dtype=torch.double, device=F_old.device)
    Cd = (F_new - F_old).T @ F_old
    Delta = torch.linalg.solve(A, Cd.T).T
    W = torch.eye(d, dtype=torch.double, device=F_old.device) + Delta
    return W


def ridge_zero_dense(F_old, F_new, lam):
    """W = solve(A, C^T)^T with C = F_new^T F_old. Verbatim from fit_drift_map
    (zero branch)."""
    F_old, F_new = F_old.double(), F_new.double()
    d = F_old.size(1)
    A = F_old.T @ F_old + lam * torch.eye(d, dtype=torch.double, device=F_old.device)
    C = F_new.T @ F_old
    W = torch.linalg.solve(A, C.T).T
    return W


# ---------------------------------------------------------------------------
# Woodbury (push-through) solvers: an (n, n) system instead of (d, d)
# ---------------------------------------------------------------------------

def _pushthrough(F_old, lam):
    """X' = (F_old F_old^T + lam I_n)^-1 F_old  (n, d)."""
    n = F_old.size(0)
    G = F_old @ F_old.T + lam * torch.eye(n, dtype=torch.double, device=F_old.device)
    return torch.linalg.solve(G, F_old)


def ridge_identity_woodbury(F_old, F_new, lam):
    """W = I + (F_new - F_old)^T (F_old F_old^T + lam I_n)^-1 F_old; equals
    ridge_identity_dense up to float64 round-off."""
    F_old, F_new = F_old.double(), F_new.double()
    d = F_old.size(1)
    X = _pushthrough(F_old, lam)
    return torch.eye(d, dtype=torch.double, device=F_old.device) + (F_new - F_old).T @ X


def ridge_zero_woodbury(F_old, F_new, lam):
    """W = F_new^T (F_old F_old^T + lam I_n)^-1 F_old."""
    F_old, F_new = F_old.double(), F_new.double()
    X = _pushthrough(F_old, lam)
    return F_new.T @ X


def solve_drift(F_old, F_new, lam, ridge="identity", solver="dense"):
    """Dispatcher: ridge in {'identity', 'zero'}, solver in {'dense', 'woodbury'}."""
    if solver not in ("dense", "woodbury"):
        raise ValueError(f"unknown drift solver {solver!r}")
    if ridge == "identity":
        fn = ridge_identity_dense if solver == "dense" else ridge_identity_woodbury
    else:
        fn = ridge_zero_dense if solver == "dense" else ridge_zero_woodbury
    return fn(F_old, F_new, lam)


# ---------------------------------------------------------------------------
# Generalised cross-validation of lam (E7b, train_drift_lambda_rule='gcv')
# ---------------------------------------------------------------------------

def gcv_scores(F_old, F_new, grid=GCV_GRID, ridge="identity"):
    """Closed-form GCV score of the multi-output ridge for every lam in grid.

    With the thin SVD X = U S V^T of X = F_old the hat matrix is
    H(lam) = U diag(h) U^T, h_i = s_i^2 / (s_i^2 + lam), shared by all d
    output coordinates, so with Z = U^T Y (Y = F_new - F_old for the identity
    anchor, F_new for the zero anchor):

        GCV(lam) = (||Y - H Y||_F^2 / n) / (1 - tr(H) / n)^2,
        ||Y - H Y||_F^2 = ||Y||_F^2 - sum_i (2 h_i - h_i^2) ||Z_i||^2,
        tr(H) = sum_i h_i.

    One SVD serves the whole grid; the current task's paired features are the
    only input (no old data, no validation split). Returns {lam: score}.
    """
    X = F_old.double()
    Y = (F_new.double() - X) if ridge == "identity" else F_new.double()
    n = X.size(0)
    U, S, _ = torch.linalg.svd(X, full_matrices=False)
    Z = U.T @ Y
    z2 = (Z ** 2).sum(dim=1)
    y2 = float((Y ** 2).sum())
    s2 = S ** 2
    out = {}
    for lam in grid:
        h = s2 / (s2 + float(lam))
        rss = max(0.0, y2 - float(((2.0 * h - h ** 2) * z2).sum()))
        denom = (1.0 - float(h.sum()) / n) ** 2
        out[float(lam)] = (rss / n) / denom if denom > 0.0 else float("inf")
    return out


def gcv_lambda(F_old, F_new, grid=GCV_GRID, ridge="identity"):
    """(lam*, scores): the grid member with the smallest GCV score (ties go
    to the first, i.e. the smallest lam of an ascending grid)."""
    scores = gcv_scores(F_old, F_new, grid, ridge)
    best = None
    for lam in grid:
        s = scores[float(lam)]
        if best is None or s < scores[best]:
            best = float(lam)
    return best, scores


# ---------------------------------------------------------------------------
# Analytic cost estimates (flop = one multiply-add counted as 2)
# ---------------------------------------------------------------------------

def flops_dense(n, d):
    """Gram X^T X (2 n d^2) + cross term Y^T X (2 n d^2) + LU of the (d, d)
    system (2/3 d^3) with d right-hand sides (2 d^3)."""
    n, d = float(n), float(d)
    return 4.0 * n * d * d + (2.0 / 3.0 + 2.0) * d ** 3


def flops_woodbury(n, d):
    """Gram X X^T (2 n^2 d) + LU of the (n, n) system (2/3 n^3) with d
    right-hand sides (2 n^2 d) + final product Y^T X' (2 n d^2)."""
    n, d = float(n), float(d)
    return 4.0 * n * n * d + (2.0 / 3.0) * n ** 3 + 2.0 * n * d * d


def flops_transport_full(K, d):
    """K classes x two (d, d) products of W S W^T."""
    return 4.0 * float(K) * float(d) ** 3


def flops_transport_diag(K, d):
    """K classes x one (d, d) matrix-vector product (W * W) v."""
    return 2.0 * float(K) * float(d) ** 2
