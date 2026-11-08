"""Check that every repo module imported by the tracked entry points is itself
tracked by git (numpy- and torch-free; runs on this PC and on the server).

Why: the rebuttal code adds sibling modules (merge_ops.py, driftsolve.py,
covstats.py, diagnostics.py, inflora_gpm.py, utils/splits.py) that main.py,
helper.py, backbones.py and utils/data_manager.py import at module import
time.  A commit or push that carries the modified entry points but not the
new modules makes `import main` fail with ModuleNotFoundError on the server,
which breaks every existing config (run_best.py, run_sota_merge.py,
run_joint.py).  This check lists exactly those modules so that they go into
the same commit; tests/smoke_rebuttal.py --tier a runs it.

How: the import statements of ENTRY_POINTS are read with `ast`; a name that
resolves to a file inside the repo (`<name>.py`, `<pkg>/<name>.py` or
`<pkg>/__init__.py`) must pass `git ls-files --error-unmatch`.  Names that do
not resolve to a repo file (third-party and standard-library modules) are
ignored.  Exit status 1 and the `git add` line to run when something is
untracked.

Usage
    python tests/check_tracked_imports.py            # from anywhere; repo root inferred
    python tests/check_tracked_imports.py --root <repo>
"""
import argparse
import ast
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: Tracked files whose imports must stay importable on a plain checkout.
ENTRY_POINTS = ("main.py", "helper.py", "backbones.py", "merging_sota.py",
                "utils/data_manager.py", "utils/data.py")


def imported_names(path):
    """Top-level dotted module names imported anywhere in a Python file."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
            # `from utils import splits` names a submodule, not an attribute
            names.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def resolve_repo_module(name, root):
    """Repo-relative path of the file that `import name` loads, or None when
    the name is not a module of this repo (third-party / stdlib)."""
    parts = name.split(".")
    candidates = (os.path.join(*parts) + ".py", os.path.join(*parts, "__init__.py"))
    for rel in candidates:
        if os.path.isfile(os.path.join(root, rel)):
            return rel.replace(os.sep, "/")
    return None


def is_tracked(rel, root):
    proc = subprocess.run(["git", "-C", root, "ls-files", "--error-unmatch", rel],
                          capture_output=True)
    return proc.returncode == 0


def find_untracked_imports(root=ROOT, entry_points=ENTRY_POINTS):
    """[(entry point, imported name, repo file)] for every imported repo module
    that git does not track.  Entry points that do not exist are skipped."""
    problems = []
    seen = set()
    for entry in entry_points:
        path = os.path.join(root, entry)
        if not os.path.isfile(path):
            continue
        for name in imported_names(path):
            rel = resolve_repo_module(name, root)
            if rel is None or rel in seen:
                continue
            seen.add(rel)
            if not is_tracked(rel, root):
                problems.append((entry, name, rel))
    return problems


def describe(problems):
    files = sorted({rel for _, _, rel in problems})
    lines = ["imported repo modules that git does not track (commit them together "
             "with the entry points that import them):"]
    lines += [f"  {entry}: import {name} -> {rel}" for entry, name, rel in problems]
    lines.append("  git add " + " ".join(files))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=ROOT, help="repository root (default: the parent of tests/)")
    args = ap.parse_args(argv)
    problems = find_untracked_imports(args.root)
    if problems:
        print(describe(problems))
        return 1
    print(f"ok: every repo module imported by {', '.join(ENTRY_POINTS)} is tracked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
