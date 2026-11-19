"""Unit checks of run_rail_fewshot's adapters against the released RAIL formulas
(numpy reference; torch on the CPU). Skipped when torch is missing.

    python tests/test_rail.py
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch
except ImportError:            # the Windows checkout has no torch
    print("SKIP tests/test_rail.py: torch not installed")
    if "pytest" in sys.modules:          # collected by pytest: skip instead of exiting
        import pytest
        pytest.skip("torch not installed", allow_module_level=True)
    sys.exit(0)

import run_rail_fewshot as rr   # noqa: E402


def np_kernel(a, b, gamma):
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)     # released utils.kernel: squared Euclidean
    return np.exp(-gamma * d)


def main():
    rng = np.random.RandomState(0)
    n, d, C = 40, 16, 4
    X = rng.randn(n, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = rng.randint(0, C, n)
    Y = np.eye(C)[y]
    Xt = torch.tensor(X)
    Yt = torch.tensor(Y)

    # dual: alpha = (K + lamda I)^-1 Y and prediction = k(x, X) alpha (released code)
    K = np_kernel(X, X, rr.GAMMA)
    alpha_ref = np.linalg.inv(K + rr.LAMDA * np.eye(n)) @ Y
    dual = rr.DualRAIL()
    dual.fit(Xt, Yt)
    assert np.allclose(dual.alpha.numpy(), alpha_ref, atol=1e-8), "dual alpha differs from the released solve"
    Xq = rng.randn(7, d); Xq /= np.linalg.norm(Xq, axis=1, keepdims=True)
    pred_ref = np_kernel(Xq, X, rr.GAMMA) @ alpha_ref
    assert np.allclose(dual.predict(torch.tensor(Xq)).numpy(), pred_ref, atol=1e-8)
    assert np.allclose(rr.rbf_kernel(Xt, Xt, rr.GAMMA).numpy(), K, atol=1e-10)

    # primal: closed form == the Woodbury form used in the code; equals the released
    # task-0 solve R @ Phi^T Y with R = (Phi^T Phi + reg I)^-1
    prim = rr.PrimalRAIL(d, hidden_dim=50, reg=rr.REGULARIZATION, seed=1)
    prim.fit(Xt, Yt)
    Phi = np.maximum(X @ prim.W.numpy(), 0.0)
    w_ref = np.linalg.inv(Phi.T @ Phi + rr.REGULARIZATION * np.eye(50)) @ Phi.T @ Y
    assert np.allclose(prim.w.numpy(), w_ref, atol=1e-8), "primal weights differ from the ridge solve"

    # recursive (RLS) update of the released code over two batches == the one-shot solve
    R = np.linalg.inv(Phi[:20].T @ Phi[:20] + rr.REGULARIZATION * np.eye(50))
    w = R @ Phi[:20].T @ Y[:20]
    P = Phi[20:]
    R = R - R @ P.T @ np.linalg.pinv(np.eye(P.shape[0]) + P @ R @ P.T) @ P @ R
    w = w + R @ P.T @ (Y[20:] - P @ w)
    assert np.allclose(w, w_ref, atol=1e-6), "RLS recursion and closed form disagree"

    # accuracy bookkeeping: a perfect predictor gives FA = AA = 100, FF = 0
    from helper import compute_metrics
    faa, ffm, ffd, asa = compute_metrics(np.full((3, 3), 100.0) * np.tri(3))
    assert faa == 100.0 and asa == 100.0 and ffm == 0.0
    print("tests/test_rail.py: 5 checks passed")


if __name__ == "__main__":
    main()
