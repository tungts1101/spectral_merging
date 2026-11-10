"""Smoke tests for the rebuttal helper modules (merge_ops, covstats,
diagnostics, driftsolve) and the default-path regression gate of the plan.

Tiers
  a   no torch (this PC): byte-compiles the modules, checks that none of
      them imports timm / peft / helper / main, checks that every repo module
      the tracked entry points import is itself tracked by git
      (tests/check_tracked_imports.py), and lists the tier-b checks.
  b   torch on the CPU (server day 0, or a CPU torch wheel): unit checks of
      the four modules; helper.pspectral_merging and merging_sota.sota_merge
      compared with the pre-rebuttal versions (`git show <ref>:<file>` loaded
      into a temporary module; the reference must differ from the worktree
      file, otherwise the comparison is vacuous and the check fails); the
      OPCM / NUFILT unit tests of tests/test_merging_sota.py; the E12
      equivalence avg(c=1) == p=1 alpha=0.5; the legacy merge_tag strings
      (imports main; skipped when main cannot be imported on this machine).
      Nothing touches the GPU, the datasets or the checkpoint cache.

Usage
    python tests/smoke_rebuttal.py --tier a
    python tests/smoke_rebuttal.py --tier b [--base-ref <commit>] [--only covstats] [--seed 0]

--base-ref defaults to PRE_REBUTTAL_REF, the last commit before the rebuttal
code (0eab95e, "Add the DIL data loaders ..."), so the gate stays meaningful
after the rebuttal code is committed (HEAD would then compare the new code
with itself).  Exit status 1 when a check fails. helper.py imports timm and
peft; where they are missing tests/_stubs.py registers stub modules first
(no-op otherwise).

Reusable helpers for other tests: make_fake_state(blocks, r, seed) builds a
LoRA state dict with the real key names and shapes (A [r, in], B [out, r],
norm.weight / norm.bias); make_fake_cache(dir, prefix, r, tasks) writes a
fake per-task chain `{prefix}_backbone_{base,0..T-1}.pt` for replay tests.
"""
import argparse
import ast
import importlib.util
import os
import py_compile
import subprocess
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

MODULES = ("merge_ops.py", "covstats.py", "diagnostics.py", "driftsolve.py", "keeplora_subspace.py")
TEST_FILES = ("tests/_stubs.py", "tests/smoke_rebuttal.py", "tests/check_tracked_imports.py")
FORBIDDEN_IMPORTS = ("timm", "peft", "helper", "main", "backbones", "merging_sota")

#: Last commit before the rebuttal code: the regression reference of the
#: helper.pspectral_merging / merging_sota.sota_merge checks (0eab95e,
#: "Add the DIL data loaders to the repo and fix .gitignore paths").
PRE_REBUTTAL_REF = "0eab95e4f6fe1523647618fa46af66fd8ff9af06"

#: LoRA target modules of the headline recipe: (key stem, in, out).
LORA_MODULES = (("attn.qkv", 768, 2304), ("mlp.fc1", 768, 3072), ("mlp.fc2", 3072, 768))


class Skip(Exception):
    """Raised by a check that cannot run in this environment."""


class Report:
    def __init__(self, only=None):
        self.rows = []
        self.only = only

    def run(self, name, fn):
        if self.only and self.only not in name:
            return
        t0 = time.time()
        try:
            msg = fn() or ""
            status = "PASS"
        except Skip as e:
            msg, status = str(e), "SKIP"
        except Exception as e:  # noqa: BLE001 - report every failure
            msg, status = f"{type(e).__name__}: {e}", "FAIL"
            traceback.print_exc()
        self.rows.append((name, status, msg))
        print(f"[{status}] {name} ({time.time() - t0:.1f}s) {msg}")

    def summary(self):
        counts = {s: sum(1 for r in self.rows if r[1] == s) for s in ("PASS", "FAIL", "SKIP")}
        print("-" * 72)
        print(f"{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped")
        for name, status, msg in self.rows:
            if status != "PASS":
                print(f"  {status}: {name}: {msg}")
        return 1 if counts["FAIL"] else 0


# ---------------------------------------------------------------------------
# Fake LoRA states and caches (shared with other tests)
# ---------------------------------------------------------------------------

def make_fake_state(blocks=12, r=64, seed=0, scale=0.02, blank_norm=True):
    """LoRA state dict with the key names and shapes of a cached
    `_backbone_t.pt` (r on qkv, fc1, fc2 of every block + Model21 norm)."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    state = {}
    for i in range(int(blocks)):
        for stem, din, dout in LORA_MODULES:
            pre = f"base_model.model.blocks.{i}.{stem}."
            state[pre + "lora_A.default.weight"] = torch.randn(r, din, generator=g) * scale
            state[pre + "lora_B.default.weight"] = torch.randn(dout, r, generator=g) * scale
    if blank_norm:
        state["norm.weight"] = torch.ones(768)
        state["norm.bias"] = torch.zeros(768)
    else:
        state["norm.weight"] = 1.0 + torch.randn(768, generator=g) * scale
        state["norm.bias"] = torch.randn(768, generator=g) * scale
    return state


def perturb(state, seed, scale=0.01):
    """state + scale * N(0, 1) noise on every tensor (new dict)."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    return {k: v + torch.randn(v.shape, generator=g) * scale for k, v in state.items()}


def make_fake_cache(ckpt_dir, prefix, r=8, tasks=3, blocks=12, seed=0):
    """Write `{prefix}_backbone_base.pt` and `{prefix}_backbone_{t}.pt` for
    t < tasks (W_t = W_{t-1} + noise) into ckpt_dir; returns the paths."""
    import torch
    os.makedirs(ckpt_dir, exist_ok=True)
    state = make_fake_state(blocks, r, seed)
    paths = []
    base = os.path.join(ckpt_dir, f"{prefix}_backbone_base.pt")
    torch.save(state, base)
    paths.append(base)
    for t in range(int(tasks)):
        state = perturb(state, seed + 1 + t, scale=0.01)
        path = os.path.join(ckpt_dir, f"{prefix}_backbone_{t}.pt")
        torch.save(state, path)
        paths.append(path)
    return paths


def states_equal(a, b):
    """True when two state dicts have the same keys, dtypes and values."""
    import torch
    if set(a) != set(b):
        return False
    return all(a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]) for k in a)


def load_git_module(relpath, ref, modname, tmpdir):
    """Load `git show <ref>:<relpath>` as a module named modname.

    Fails (AssertionError) when the reference blob equals the worktree file:
    the regression check would then compare the code with itself, which is
    what happens with --base-ref HEAD once the rebuttal code is committed."""
    proc = subprocess.run(["git", "-C", ROOT, "show", f"{ref}:{relpath}"],
                          capture_output=True)
    if proc.returncode != 0:
        raise Skip(f"git show {ref}:{relpath} failed: "
                   f"{proc.stderr.decode('utf-8', 'replace').strip()}")
    source = proc.stdout.decode("utf-8")
    with open(os.path.join(ROOT, relpath), encoding="utf-8") as f:
        current = f.read()
    if source.replace("\r\n", "\n") == current.replace("\r\n", "\n"):
        raise AssertionError(
            f"{ref}:{relpath} is identical to the worktree file, so the regression "
            f"comparison is vacuous; pass --base-ref <pre-rebuttal commit> "
            f"(default PRE_REBUTTAL_REF = {PRE_REBUTTAL_REF[:7]})")
    path = os.path.join(tmpdir, modname + ".py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def legacy_tag(cfg):
    """The merged-checkpoint tag of main.py before merge_tag existed
    (Learner.merged_checkpoint and the run_config_sweep cleanup block)."""
    method = cfg["train_merge"]
    variant = cfg.get("spectral_variant", cfg.get("pspectral_variant", "only_A"))
    if method == "pspectral":
        return f"pspec_{variant}_p{cfg['pspectral_p']}_a{cfg['train_merge_alpha']}"
    if method == "kspectral":
        return f"kspec_{variant}_k{cfg['train_merge_k']}_a{cfg['train_merge_alpha']}"
    return f"{method}_c{cfg['train_merge_coef']}"


# ---------------------------------------------------------------------------
# Tier a: static checks (no torch)
# ---------------------------------------------------------------------------

TIER_B_CHECKS = (
    "merge_ops: lora_pairs / pair_label",
    "merge_ops: energy_rank == helper.truncate_energy rank",
    "merge_ops: spectral_step trunc / clip / scale_fro",
    "merge_ops: alpha_from_stat, effective_update, step_norms",
    "merge_ops: fake cache replay (E4 shape)",
    "helper: pspectral_merging regression vs git ref",
    "helper: avg(c=1) == pspectral(p=1, alpha=0.5) (E12)",
    "merging_sota: sota_merge regression vs git ref",
    "merging_sota: opcm / nufilt unit tests (tests/test_merging_sota.py)",
    "covstats: fit / transport / make_dist / storage",
    "diagnostics: gaussian_errors / transport_gaussian / csv_append",
    "driftsolve: dense == woodbury, verbatim dense, GCV, flops",
    "main: merge_tag legacy strings",
)


def tier_a(report):
    def compile_all():
        for rel in MODULES + TEST_FILES:
            py_compile.compile(os.path.join(ROOT, rel), doraise=True)
        return f"{len(MODULES) + len(TEST_FILES)} files"

    def no_forbidden_imports():
        bad = []
        for rel in MODULES:
            with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=rel)
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for n in names:
                    if n.split(".")[0] in FORBIDDEN_IMPORTS:
                        bad.append(f"{rel}: {n}")
        if bad:
            raise AssertionError("forbidden imports: " + ", ".join(bad))
        return "torch-only modules"

    def tracked_imports():
        # Every repo module imported by main.py, helper.py, backbones.py,
        # merging_sota.py and utils/data_manager.py must be tracked by git:
        # a commit of the modified entry points without the new modules would
        # break `import main` on the server (tests/check_tracked_imports.py).
        import check_tracked_imports
        problems = check_tracked_imports.find_untracked_imports(ROOT)
        if problems:
            raise AssertionError(check_tracked_imports.describe(problems))
        return "every imported repo module is tracked"

    def list_tier_b():
        print("  checks that need torch (run `--tier b` on the server CPU):")
        for c in TIER_B_CHECKS:
            print(f"    - {c}")
        return f"{len(TIER_B_CHECKS)} checks deferred"

    report.run("tier-a: py_compile", compile_all)
    report.run("tier-a: no timm/peft/helper/main imports", no_forbidden_imports)
    report.run("tier-a: imported repo modules are git-tracked", tracked_imports)
    report.run("tier-a: tier-b inventory", list_tier_b)


# ---------------------------------------------------------------------------
# Tier b: torch checks
# ---------------------------------------------------------------------------

def _import_helper():
    try:
        import helper  # noqa: F401
        return helper
    except Exception as e:  # noqa: BLE001
        raise Skip(f"import helper failed even with stubs: {e!r}")


def check_lora_pairs():
    import torch
    import merge_ops as mo
    st = make_fake_state(blocks=12, r=64, seed=0)
    pairs = mo.lora_pairs(st)
    assert len(pairs) == 36, len(pairs)
    for a, b in pairs:
        assert a.endswith("lora_A.default.weight") and b.endswith("lora_B.default.weight")
        assert mo.pair_prefix(a) == mo.pair_prefix(b)
        assert st[a].shape[0] == st[b].shape[1] == 64
    # same order as helper.pspectral_merging's detection loop
    ref = {}
    for name in st:
        for a_suf, b_suf in mo.FACTOR_SUFFIXES:
            if name.endswith(a_suf):
                ref.setdefault(name[: -len(a_suf)], {})["A"] = name
            elif name.endswith(b_suf):
                ref.setdefault(name[: -len(b_suf)], {})["B"] = name
    assert pairs == [(d["A"], d["B"]) for d in ref.values()]
    # AdaptFormer suffixes, incomplete pairs, labels
    st2 = {"x.down_proj.weight": torch.zeros(4, 8), "x.up_proj.weight": torch.zeros(8, 4),
           "x.bias": torch.zeros(8), "y.lora_A.default.weight": torch.zeros(2, 2)}
    assert mo.lora_pairs(st2) == [("x.down_proj.weight", "x.up_proj.weight")]
    assert mo.pair_label("base_model.model.blocks.3.mlp.fc1.lora_A.default.weight") == (3, "fc1")
    assert mo.pair_label("base_model.model.blocks.11.attn.qkv.lora_B.default.weight") == (11, "qkv")
    assert mo.pair_label("x.down_proj.weight") == (None, "x")
    assert mo.lora_scale(128, 64) == 2.0
    return "36 pairs, helper order"


def check_energy_rank():
    import torch
    import merge_ops as mo
    helper = _import_helper()
    g = torch.Generator().manual_seed(1)
    ps = (0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0)
    n_checked = 0
    for shape in ((64, 768), (64, 3072), (16, 768), (8, 768)):
        d = torch.randn(shape, generator=g)
        prev = 0
        for p in ps:
            k, S, cum = mo.energy_rank(d, p)
            assert 1 <= k <= min(shape) and k >= prev, (shape, p, k, prev)
            prev = k
            rec = helper.truncate_energy(d, p)
            assert int(torch.linalg.matrix_rank(rec)) == k, (shape, p, k)
            assert torch.equal(mo.truncate_energy(d, p), rec)
            # the rule itself: cum[k-2] < p <= cum[k-1] (or clamped at full rank)
            if k >= 2:
                assert float(cum[k - 2]) < p
            if k < S.numel():
                assert float(cum[k - 1]) >= p
            assert mo.rank_from_svals(S, p) == k
            n_checked += 1
        assert mo.energy_rank(d, 1.0)[0] == min(shape)
    # low-rank update: k never exceeds the true rank at p < 1
    low = torch.randn(64, 5, generator=g) @ torch.randn(5, 768, generator=g)
    assert mo.energy_rank(low, 0.9)[0] <= 5
    assert mo.energy_rank(low, 0.999)[0] <= 5
    # degenerate updates
    assert mo.energy_rank(torch.zeros(4, 6), 0.9)[0] == 0
    assert mo.rank_from_svals(torch.zeros(0), 0.9) == 0
    S = mo.energy_rank(torch.randn(8, 32, generator=g), 0.9)[1]
    assert 0.0 < mo.energy_share(S, 1) < 1.0 and abs(mo.energy_share(S, 8) - 1.0) < 1e-6
    return f"{n_checked} (shape, p) cells equal"


def check_spectral_step():
    import torch
    import merge_ops as mo
    helper = _import_helper()
    g = torch.Generator().manual_seed(2)
    d = torch.randn(64, 768, generator=g)
    for p, alpha in ((0.9, 0.3), (0.5, 1.0), (1.0, 0.15), (0.9, 1.0)):
        assert torch.equal(mo.spectral_step(d, p, alpha, "trunc"),
                           alpha * helper.truncate_energy(d, p))
    S = torch.linalg.svdvals(d)
    for alpha in (0.3, 0.5, 1.0):
        step = mo.spectral_step(d, 0.9, alpha, "clip")
        assert step.shape == d.shape and step.dtype == d.dtype
        Sc = torch.linalg.svdvals(step)
        assert abs(float(Sc[0]) - alpha * float(S[0])) <= 1e-5 * float(S[0]), (alpha, Sc[0], S[0])
        tail = S < alpha * S[0]
        assert torch.allclose(Sc[tail], S[tail], rtol=1e-4, atol=1e-5)
        assert int(torch.linalg.matrix_rank(step)) == 64  # every direction kept
        if alpha == 1.0:
            assert torch.allclose(step, d, atol=1e-5)
    for p, alpha in ((0.9, 0.3), (0.5, 0.5), (1.0, 0.3)):
        step = mo.spectral_step(d, p, alpha, "scale_fro")
        target = alpha * float(helper.truncate_energy(d, p).norm())
        assert abs(float(step.norm()) - target) <= 1e-5 * target, (p, alpha)
        beta = float(step.norm()) / float(d.norm())
        assert torch.allclose(step, beta * d, rtol=1e-5, atol=1e-6)
        if p == 1.0:
            assert torch.allclose(step, alpha * d, rtol=1e-5, atol=1e-6)
    z = torch.zeros(4, 6)
    for op in mo.UPDATE_OPS:
        assert torch.equal(mo.spectral_step(z, 0.9, 0.3, op), z)
    try:
        mo.spectral_step(d, 0.9, 0.3, "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown op must raise ValueError")
    return "trunc equal, clip/scale_fro norms matched"


def check_alpha_and_norms():
    import torch
    import merge_ops as mo
    a = mo.alpha_from_stat
    assert a(2.0, 1.0) == 0.5
    assert a(100.0, 1.0, 0.05) == 0.05
    assert a(0.1, 1.0) == 1.0
    assert a(0.0, 1.0) == 1.0 and a(float("nan"), 1.0) == 1.0 and a(-1.0, 1.0) == 1.0
    vals = [a(s, 1.0) for s in (0.5, 1.0, 2.0, 4.0, 8.0, 40.0)]
    assert all(x >= y for x, y in zip(vals, vals[1:])), vals
    w0 = make_fake_state(blocks=2, r=16, seed=3)
    w1 = perturb(w0, seed=4)
    pairs = mo.lora_pairs(w1)
    for an, bn in pairs:
        eff = mo.effective_update(w0, w1, an, bn, 2.0)
        ref = 2.0 * (w1[bn].float() @ w1[an].float() - w0[bn].float() @ w0[an].float())
        assert torch.equal(eff, ref)
    sn = mo.step_norms(w0, w1, scale=2.0, p=0.9)
    agg = sn["agg"]
    assert agg["n_pairs"] == 6 and len(sn["pairs"]) == 6
    for key in ("k_mean", "k_min", "k_max", "sigma1_A_mean", "sigma1_A_max",
                "fro_A_mean", "sigma1_eff_mean", "sigma1_eff_max", "fro_eff_mean"):
        assert key in agg, key
    assert 1 <= agg["k_min"] <= agg["k_mean"] <= agg["k_max"] <= 16
    an, bn = pairs[0]
    row = sn["pairs"][mo.pair_prefix(an)]
    s1 = float(torch.linalg.svdvals(mo.effective_update(w0, w1, an, bn, 2.0))[0])
    assert abs(row["sigma1_eff"] - s1) <= 1e-4 * s1
    assert abs(row["fro_A"] - float((w1[an] - w0[an]).norm())) <= 1e-5
    zero = mo.step_norms(w0, w0, scale=2.0)
    assert all(v["k"] == 0 and v["sigma1_A"] == 0.0 for v in zero["pairs"].values())
    assert mo.step_norms({}, {}, 2.0)["agg"] == {"n_pairs": 0}
    return "rule clipped and monotone; norms consistent"


def check_fake_cache_replay(tmpdir):
    import torch
    import merge_ops as mo
    helper = _import_helper()
    cache = os.path.join(tmpdir, "fake_cache")
    prefix = "1993_cifar224_vit_base_patch16_224_lora_smoke_r8"
    paths = make_fake_cache(cache, prefix, r=8, tasks=3, seed=11)
    assert len(paths) == 4 and all(os.path.exists(p) for p in paths)
    states = [torch.load(os.path.join(cache, f"{prefix}_backbone_{t}.pt"), map_location="cpu")
              for t in range(3)]
    assert len(mo.lora_pairs(states[0])) == 36 and "norm.weight" in states[0]
    assert states[0]["base_model.model.blocks.0.attn.qkv.lora_A.default.weight"].shape == (8, 768)
    ps = (0.5, 0.7, 0.9, 0.95, 0.99)
    rows = []
    M = states[0]
    for t in (1, 2):
        for an, bn in mo.lora_pairs(states[t]):
            dA = states[t][an].float() - M[an].float()
            ks = [mo.energy_rank(dA, p)[0] for p in ps]
            assert all(1 <= k <= 8 for k in ks) and ks == sorted(ks), ks
            rows.append((t, an, ks))
        M = helper.pspectral_merging(M, states[t], p=0.9, alpha=0.3, variant="only_A")
    assert len(rows) == 72
    return "36 pairs x 2 tasks, 1 <= k <= r, k monotone in p"


def check_pspectral_regression(tmpdir, ref):
    import torch  # noqa: F401
    helper = _import_helper()
    head = load_git_module("helper.py", ref, "helper_ref", tmpdir)
    w0 = make_fake_state(blocks=3, r=16, seed=5, blank_norm=False)
    w1 = perturb(w0, seed=6)
    assert states_equal(helper.pspectral_merging(w0, w1), head.pspectral_merging(w0, w1)), \
        "default kwargs differ"
    n = 1
    for variant in ("only_A", "only_B", "separate", "combine"):
        for p, alpha in ((0.9, 0.3), (0.5, 1.0), (1.0, 0.15)):
            a = helper.pspectral_merging(w0, w1, p=p, alpha=alpha, variant=variant)
            b = head.pspectral_merging(w0, w1, p=p, alpha=alpha, variant=variant)
            assert states_equal(a, b), f"{variant} p={p} alpha={alpha} differs from {ref}"
            n += 1
    # production shapes at the headline posture (only_A, p 0.9, r 64, 36 pairs)
    w0 = make_fake_state(blocks=12, r=64, seed=7)
    w1 = perturb(w0, seed=8)
    a = helper.pspectral_merging(w0, w1, p=0.9, alpha=0.3, variant="only_A")
    b = head.pspectral_merging(w0, w1, p=0.9, alpha=0.3, variant="only_A")
    assert states_equal(a, b), f"headline only_A differs from {ref}"
    assert set(a) == set(w1)
    return f"{n + 1} calls torch.equal to {ref}:helper.py"


def check_avg_equivalence():
    import torch
    helper = _import_helper()
    base = make_fake_state(blocks=2, r=16, seed=9)
    M = perturb(base, seed=10)
    W = perturb(base, seed=11)
    avg = helper.merge(base, [M, W], "avg", lamb=1.0)
    for k in avg:
        assert torch.allclose(avg[k], (M[k] + W[k]) / 2, atol=1e-6, rtol=0), k
    ps = helper.pspectral_merging(M, W, p=1.0, alpha=0.5, variant="only_A")
    maxdev = max(float((ps[k] - (M[k] + W[k]) / 2).abs().max()) for k in ps)
    scale = max(float(((M[k] + W[k]) / 2).abs().max()) for k in ps)
    assert maxdev <= 1e-4 * max(1.0, scale), maxdev
    return f"max |pspectral(p=1, a=0.5) - (M+W)/2| = {maxdev:.2e} (float32 SVD round-off)"


def check_sota_regression(tmpdir, ref):
    import torch
    import merging_sota as ms
    head = load_git_module("merging_sota.py", ref, "merging_sota_ref", tmpdir)
    assert tuple(ms.SOTA_METHODS[:5]) == ("dare", "della", "breadcrumbs", "knots", "modelstock")
    base = make_fake_state(blocks=2, r=16, seed=12)
    M = perturb(base, seed=13)
    W = perturb(base, seed=14)
    kwargs = dict(lamb=1.0, dare_q=0.5, della_q=0.5, bc_beta=0.5, bc_gamma=0.01, knots_topk=100.0)
    for method in ("dare", "della", "breadcrumbs", "knots", "modelstock"):
        torch.manual_seed(123)
        a = ms.sota_merge(base, M, W, method, **kwargs)
        torch.manual_seed(123)
        b = head.sota_merge(base, M, W, method, **kwargs)
        assert states_equal(a, b), f"{method} differs from {ref}:merging_sota.py"
    for mod in (ms, head):
        try:
            mod.sota_merge(base, M, W, "bogus")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown sota method must raise ValueError")
    return f"5 rules torch.equal to {ref}:merging_sota.py"


def check_covstats(seed):
    import torch
    import covstats as cs
    torch.manual_seed(seed)
    d, n = 32, 400
    L = torch.randn(d, d) * 0.3
    Sigma = L @ L.T + 0.5 * torch.eye(d)
    mu = torch.randn(d) * 2
    X = mu + torch.randn(n, d) @ torch.linalg.cholesky(Sigma).T
    mean, cov = cs.fit_full(X)
    var = cs.fit_diag(X)
    U, vres = cs.fit_lowrank(X, 8)
    Sc = cs.scatter(X)
    assert mean.shape == (d,) and cov.shape == (d, d) and U.shape == (d, 8)
    assert torch.allclose(cov.diagonal(), var, atol=1e-4)
    assert torch.allclose((U @ U.T + torch.diag(vres)).diagonal(), cov.diagonal(), atol=1e-4)
    assert torch.allclose(cs.shared_cov(Sc, n - 1), cov, atol=1e-4)
    Uf, vf = cs.fit_lowrank(X, d)
    relF = float((Uf @ Uf.T + torch.diag(vf) - cov).norm() / cov.norm())
    assert relF < 1e-4, relF
    m1, c1 = cs.fit_full(X[:1])
    assert torch.equal(m1, X[0]) and torch.allclose(c1, torch.eye(d) * 1e-4)
    # transport
    W = torch.eye(d) + 0.1 * torch.randn(d, d) / d ** 0.5
    Wd = W.double()
    ref = torch.diag(Wd @ torch.diag(var.double()) @ Wd.T).float()
    assert torch.allclose(cs.transport_diag(var, W), ref, rtol=1e-5, atol=1e-7)
    both = cs.transport_diag(torch.stack([var, 2 * var]), W)
    assert torch.allclose(both[1], 2 * ref, rtol=1e-5, atol=1e-7)
    covs = torch.stack([cov, cov * 1.5, cov + 0.1 * torch.eye(d)])
    loop = []
    for c in covs:                       # transport_prototypes, main.py:1234-1237
        S = Wd @ c.double() @ Wd.T
        S = 0.5 * (S + S.T) + torch.eye(d, dtype=torch.double) * 1e-4
        loop.append(S.float())
    loop = torch.stack(loop)
    assert torch.allclose(cs.transport_full(covs, W), loop, atol=1e-6, rtol=1e-6)
    assert torch.allclose(cs.transport_full(cov, W), loop[0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(cs.transport_full(Sc, W, jitter=0.0), Wd @ Sc @ Wd.T, rtol=1e-10)
    lhs = torch.diag(Wd @ (U.double() @ U.double().T + torch.diag(vres.double())) @ Wd.T)
    WU = cs.transport_lowrank_U(U.double(), Wd)
    rhs = (WU ** 2).sum(1) + cs.transport_diag(vres.double(), Wd)
    assert torch.allclose(lhs, rhs, rtol=1e-10)
    assert cs.transport_lowrank_U(U, W).dtype == torch.float32
    assert cs.transport_lowrank_U(torch.stack([U, U]), W).shape == (2, d, 8)
    # distributions
    L_sh = cs.shared_scale_tril(Sc, n - 1)
    shared = cs.shared_cov(Sc, n - 1)
    assert torch.allclose(L_sh @ L_sh.T, shared, atol=1e-4)
    dists = {
        "class": cs.make_dist("class", mean, cov=cov),
        "diag": cs.make_dist("diag", mean, var=var),
        "lowrank": cs.make_dist("lowrank", mean, U=U, var=vres),
        "shared": cs.make_dist("shared", mean, shared_L=L_sh),
    }
    targets = {
        "class": cs.to_full_cov("class", cov=cov),
        "diag": cs.to_full_cov("diag", var=var),
        "lowrank": cs.to_full_cov("lowrank", U=U, var=vres),
        "shared": cs.to_full_cov("shared", shared=shared),
    }
    for mode, dist in dists.items():
        s = dist.sample((5,))
        assert s.shape == (5, d), (mode, s.shape)
        big = dist.sample((20000,))
        target = targets[mode]
        rel = float((torch.cov(big.T) - target).norm() / target.norm())
        assert rel < 0.15, (mode, rel)
        assert float((big.mean(0) - mean).norm() / mean.norm()) < 0.05, mode
    assert torch.equal(cs.to_full_cov("diag", var=var), torch.diag(var))
    # storage accounting
    assert cs.storage_floats("class", 768, 1) == 590_592
    assert cs.storage_floats("full_sym", 768, 1) == 296_064
    assert cs.storage_floats("diag", 768, 1) == 1_536
    assert cs.storage_floats("lowrank", 768, 1, 16) == 13_824
    assert cs.storage_floats("lowrank", 768, 1, 64) == 50_688
    assert cs.storage_floats("shared", 768, 200) == 200 * 768 + 768 ** 2
    assert cs.storage_floats("class", 768, 100) == 59_059_200
    assert cs.lora_param_count(64) == 8_259_072
    assert cs.lora_param_count(64, norm=0) == 129_024 * 64
    assert cs.lora_param_count(8, norm=0) == 129_024 * 8
    for bad in (lambda: cs.make_dist("bogus", mean), lambda: cs.storage_floats("bogus", 768, 1),
                lambda: cs.to_full_cov("bogus"), lambda: cs.make_dist("diag", mean)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid mode / missing argument must raise ValueError")
    return "4 modes fit, transported and sampled; storage numbers match"


def check_diagnostics(seed, tmpdir):
    import torch
    import covstats as cs
    import diagnostics as dg
    import driftsolve as ds
    torch.manual_seed(seed)
    d, n = 16, 4000
    A = torch.randn(d, d) * 0.5
    mu = torch.randn(d) * 2
    X = mu + torch.randn(n, d) @ A.T
    W = torch.eye(d) + 0.1 * torch.randn(d, d) / 4
    Y = X @ W.T + 0.01 * torch.randn(n, d)
    muX, SX = cs.fit_full(X)
    muY, SY = cs.fit_full(Y)
    muT, ST = dg.transport_gaussian(muX, SX, W)
    # same arithmetic as transport_prototypes
    Wd = W.double()
    assert torch.equal(muT, (muX.double() @ Wd.T).float())
    S = Wd @ SX.double() @ Wd.T
    S = 0.5 * (S + S.T) + torch.eye(d, dtype=torch.double) * 1e-4
    assert torch.equal(ST, S.float())
    err_t = dg.gaussian_errors(muT, ST, muY, SY, kl=True)
    err_0 = dg.gaussian_errors(muX, SX, muY, SY, kl=True)
    assert err_t["mean_rel"] < 0.02 and err_t["cov_relF"] < 0.1, err_t
    assert err_0["mean_rel"] > err_t["mean_rel"] and err_0["cov_relF"] > err_t["cov_relF"]
    assert err_0["symkl"] > err_t["symkl"] >= -1e-8
    same = dg.gaussian_errors(muX, SX, muX, SX, kl=True)
    assert same["mean_rel"] == 0.0 and same["cov_relF"] == 0.0 and abs(same["symkl"]) < 1e-8
    assert abs(same["mean_cos"] - 1.0) < 1e-9 and same["trace_ratio"] == 1.0
    W_or = ds.ridge_identity_dense(X, Y, 1e-3)
    assert float((W_or.float() - W).abs().max()) < 1e-2
    gm = dg.group_means([{"g": "x", "v": 1.0, "flag": True}, {"g": "x", "v": 3.0},
                         {"g": "y", "v": 5.0, "s": "txt"}], "g")
    assert gm["x"]["v"] == 2.0 and gm["x"]["n"] == 2 and gm["y"]["v"] == 5.0
    assert "flag" not in gm["x"] and "s" not in gm["y"]
    assert dg.group_means([{"v": 2.0}, {"v": 4.0}])["all"]["v"] == 3.0
    path = os.path.join(tmpdir, "csv", "diag.csv")
    dg.csv_append(path, ["a", "b"], [{"a": 1, "b": 2.5}])
    dg.csv_append(path, ["a", "b"], [[3, 4]])
    with open(path, encoding="utf-8") as f:
        assert f.read().splitlines() == ["a,b", "1,2.5", "3,4"]
    return (f"transported mean_rel {err_t['mean_rel']:.1e} vs uncorrected "
            f"{err_0['mean_rel']:.2f}; header written once")


def check_driftsolve(seed):
    import torch
    import covstats as cs
    import driftsolve as ds
    torch.manual_seed(seed)
    grid = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
    for n, d in ((50, 80), (120, 80)):
        X = torch.randn(n, d).double()
        T = torch.eye(d, dtype=torch.double) + 0.05 * torch.randn(d, d).double() / d ** 0.5
        Y = X @ T.T + 0.01 * torch.randn(n, d).double()
        I = torch.eye(d, dtype=torch.double)
        for lam in (1.0, 100.0):
            Wd = ds.ridge_identity_dense(X, Y, lam)
            Ww = ds.ridge_identity_woodbury(X, Y, lam)
            assert float((Wd - Ww).abs().max()) < 1e-10, (n, d, lam)
            Zd = ds.ridge_zero_dense(X, Y, lam)
            Zw = ds.ridge_zero_woodbury(X, Y, lam)
            assert float((Zd - Zw).abs().max()) < 1e-10, (n, d, lam)
            # verbatim transcription of main.py:1205-1212
            A = X.T @ X + lam * torch.eye(d, dtype=torch.double)
            Cd = (Y - X).T @ X
            Delta = torch.linalg.solve(A, Cd.T).T
            assert torch.equal(torch.eye(d, dtype=torch.double) + Delta, Wd)
            C = Y.T @ X
            assert torch.equal(torch.linalg.solve(A, C.T).T, Zd)
            assert torch.equal(ds.solve_drift(X, Y, lam, "identity", "dense"), Wd)
            assert torch.equal(ds.solve_drift(X, Y, lam, "zero", "woodbury"), Zw)
            assert Wd.dtype == torch.float64
            # float32 inputs are promoted
            assert torch.allclose(ds.ridge_identity_dense(X.float(), Y.float(), lam), Wd, atol=1e-3)
        # GCV: closed form equals the explicit hat matrix and the ridge residual
        scores = ds.gcv_scores(X, Y, grid)
        Yc = Y - X
        for lam in grid:
            H = X @ torch.linalg.solve(X.T @ X + lam * I, X.T)
            R = Yc - H @ Yc
            rss = float((R ** 2).sum())
            df = float(torch.trace(H))
            ref = (rss / n) / (1.0 - df / n) ** 2
            assert abs(scores[lam] - ref) <= 1e-8 * max(1.0, ref), (n, lam, scores[lam], ref)
            Delta = ds.ridge_identity_dense(X, Y, lam) - I
            rss2 = float(((Yc - X @ Delta.T) ** 2).sum())
            assert abs(rss - rss2) <= 1e-8 * max(1.0, rss)
        best, sc = ds.gcv_lambda(X, Y, grid)
        assert best in grid and sc[best] == min(sc.values()) and all(v > 0 for v in sc.values())
        z = ds.gcv_scores(X, Y, grid, ridge="zero")
        assert all(v > 0 for v in z.values())
    try:
        ds.solve_drift(X, Y, 1.0, "identity", "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown solver must raise ValueError")
    assert ds.flops_woodbury(480, 768) < ds.flops_dense(480, 768)
    assert ds.flops_dense(5000, 768) < ds.flops_woodbury(5000, 768)
    assert ds.flops_transport_diag(100, 768) < ds.flops_transport_full(100, 768)
    assert ds.transport_full is cs.transport_full and ds.transport_diag is cs.transport_diag
    assert ds.GCV_GRID == grid
    return "dense == woodbury (<1e-10), verbatim dense, GCV closed form verified"


def check_merge_tag():
    try:
        import main
    except Exception as e:  # noqa: BLE001
        raise Skip(f"import main failed on this machine: {e!r}")
    if not hasattr(main, "merge_tag"):
        raise AssertionError("main.merge_tag is missing (integration step 1 of the plan)")
    cases = [
        ({"train_merge": "pspectral", "spectral_variant": "only_A", "pspectral_p": 0.9,
          "train_merge_alpha": 0.3}, "pspec_only_A_p0.9_a0.3"),
        ({"train_merge": "pspectral", "spectral_variant": "only_A", "pspectral_p": 0.9,
          "train_merge_alpha": 0.15}, "pspec_only_A_p0.9_a0.15"),
        ({"train_merge": "pspectral", "spectral_variant": "combine", "pspectral_p": 1.0,
          "train_merge_alpha": 1.0}, "pspec_combine_p1.0_a1.0"),
        ({"train_merge": "pspectral", "spectral_variant": "only_A", "pspectral_p": 1,
          "train_merge_alpha": 0.1}, "pspec_only_A_p1_a0.1"),
        ({"train_merge": "pspectral", "pspectral_variant": "separate", "pspectral_p": 0.5,
          "train_merge_alpha": 0.7, "_drop": ("spectral_variant",)},
         "pspec_separate_p0.5_a0.7"),
        ({"train_merge": "pspectral", "pspectral_p": 0.9, "train_merge_alpha": 0.2,
          "_drop": ("spectral_variant", "pspectral_variant")}, "pspec_only_A_p0.9_a0.2"),
        ({"train_merge": "kspectral", "spectral_variant": "only_B", "train_merge_k": 16,
          "train_merge_alpha": 0.3}, "kspec_only_B_k16_a0.3"),
        ({"train_merge": "avg", "train_merge_coef": 1.0}, "avg_c1.0"),
        ({"train_merge": "ties", "train_merge_coef": 1.2}, "ties_c1.2"),
        ({"train_merge": "dare", "train_merge_coef": 0.5}, "dare_c0.5"),
        ({"train_merge": "knots", "train_merge_coef": 1}, "knots_c1"),
        ({"train_merge": "none", "train_merge_coef": 1.0}, "none_c1.0"),
    ]
    for over, expected in cases:
        cfg = dict(main.BASE_CONFIG)
        cfg.update({k: v for k, v in over.items() if k != "_drop"})
        for k in over.get("_drop", ()):
            cfg.pop(k, None)
        assert legacy_tag(cfg) == expected, ("test fixture", expected, legacy_tag(cfg))
        got = main.merge_tag(cfg)
        assert got == expected, f"merge_tag -> {got!r}, legacy {expected!r}"
    base = dict(main.BASE_CONFIG, train_merge="pspectral", spectral_variant="only_A",
                pspectral_p=0.9, train_merge_alpha=0.3)
    legacy = legacy_tag(base)
    assert main.merge_tag(dict(base, merge_update_op="trunc", merge_alpha_rule="fixed")) == legacy
    notes = []
    for over, token, planned in (
        ({"merge_update_op": "clip"}, "clip", "pspec_only_A_p0.9_clip_a0.3"),
        ({"merge_alpha_rule": "specnorm", "merge_alpha_param": 0.5}, "specnorm",
         "pspec_only_A_p0.9_aspecnorm0.5"),
    ):
        tag = main.merge_tag(dict(base, **over))
        assert tag != legacy and token in tag, f"{over} -> {tag!r} collides with {legacy!r}"
        if tag != planned:
            notes.append(f"WARN {over} -> {tag!r} (plan S1 wrote {planned!r})")
    return f"{len(cases)} legacy tags identical" + ("; " + "; ".join(notes) if notes else "")


def tier_b(report, args):
    try:
        import torch
    except ImportError:
        print("torch is not installed: tier b needs a CPU torch wheel "
              "(pip install torch --index-url https://download.pytorch.org/whl/cpu). "
              "All tier-b checks SKIPPED.")
        for name in TIER_B_CHECKS:
            report.rows.append((name, "SKIP", "torch missing"))
        return
    import _stubs
    stubbed = _stubs.install()
    os.chdir(ROOT)   # `import main` creates logs_main/ relative to the cwd
    # Scratch directory: the caller's --tmp-dir (kept), else a temporary
    # directory removed at the end (nothing is left in the system temp).
    scratch = None if args.tmp_dir else tempfile.TemporaryDirectory(prefix="smoke_rebuttal_")
    tmpdir = args.tmp_dir or scratch.name
    os.makedirs(tmpdir, exist_ok=True)
    print(f"torch {torch.__version__} on CPU; stubbed: {stubbed or 'nothing'}; tmp: {tmpdir}; "
          f"regression reference {args.base_ref[:12]}")
    torch.manual_seed(args.seed)

    try:
        report.run("merge_ops: lora_pairs / pair_label", check_lora_pairs)
        report.run("merge_ops: energy_rank == helper.truncate_energy rank", check_energy_rank)
        report.run("merge_ops: spectral_step trunc / clip / scale_fro", check_spectral_step)
        report.run("merge_ops: alpha_from_stat, effective_update, step_norms", check_alpha_and_norms)
        report.run("merge_ops: fake cache replay (E4 shape)", lambda: check_fake_cache_replay(tmpdir))
        report.run("helper: pspectral_merging regression vs git ref",
                   lambda: check_pspectral_regression(tmpdir, args.base_ref))
        report.run("helper: avg(c=1) == pspectral(p=1, alpha=0.5) (E12)", check_avg_equivalence)
        report.run("merging_sota: sota_merge regression vs git ref",
                   lambda: check_sota_regression(tmpdir, args.base_ref))
        report.run("merging_sota: opcm / nufilt unit tests (tests/test_merging_sota.py)",
                   lambda: check_merging_sota_tests(args.base_ref))
        report.run("covstats: fit / transport / make_dist / storage", lambda: check_covstats(args.seed))
        report.run("diagnostics: gaussian_errors / transport_gaussian / csv_append",
                   lambda: check_diagnostics(args.seed, tmpdir))
        report.run("driftsolve: dense == woodbury, verbatim dense, GCV, flops",
                   lambda: check_driftsolve(args.seed))
        report.run("main: merge_tag legacy strings", check_merge_tag)
    finally:
        if scratch is not None:
            try:
                scratch.cleanup()
            except OSError as exc:           # a lingering handle must not fail the gate
                print(f"(could not remove {tmpdir}: {exc})")


def check_merging_sota_tests(ref):
    """Run the unit tests of tests/test_merging_sota.py (OPCM / NUFILT math,
    dispatch, and its own regression test against `ref`) in this process."""
    import test_merging_sota as tms
    os.environ["MERGING_SOTA_REF"] = ref
    for fn in tms.TESTS:
        fn()
    return f"{len(tms.TESTS)} tests passed (reference {ref[:12]})"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tier", choices=("a", "b"), required=True)
    ap.add_argument("--base-ref", default=PRE_REBUTTAL_REF,
                    help="git ref whose helper.py / merging_sota.py are the regression reference "
                         "(default: the last pre-rebuttal commit %(default).7s)")
    ap.add_argument("--only", default=None, help="run only the checks whose name contains this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmp-dir", default=None,
                    help="scratch directory to keep (default: a temporary directory removed at the end)")
    args = ap.parse_args()
    report = Report(only=args.only)
    if args.tier == "a":
        tier_a(report)
    else:
        tier_b(report, args)
    sys.exit(report.summary())


if __name__ == "__main__":
    main()
